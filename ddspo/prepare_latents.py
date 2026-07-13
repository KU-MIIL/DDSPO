#!/usr/bin/env python
# coding=utf-8
"""Pre-compute the paired latents consumed by DDSPO training.

For every prompt in the input JSONL, a positive sample is generated from the
original prompt and a negative sample from a (randomly chosen) semantically
degraded prompt, both with the pretrained pipeline. The VAE latents are written
to ``<save_dir>/latents/<id>{,_neg}.safetensors`` and a ``metadata.jsonl`` is
produced in the layout expected by ``ddspo/data.py``.

Input JSONL lines::

    {"id": "...", "prompt": "...", "neg_prompts": ["...", ...], "tag": "..."}

Launch with ``accelerate launch`` for multi-GPU generation.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

MODEL_TYPES = ["sd15", "sdxl", "sd3", "sana"]


def _collate(batch):
    return list(zip(*batch))


class JsonDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = [json.loads(line) for line in f if json.loads(line).get("neg_prompts")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        neg_prompts = item["neg_prompts"]
        return (item["id"], item.get("tag", ""), item["prompt"],
                random.choice(neg_prompts), neg_prompts)


def _paths(save_dir, save_type, sample_id):
    sub, ext = ("latents", "safetensors") if save_type == "latent" else ("images", "png")
    return (os.path.join(save_dir, sub, f"{sample_id}.{ext}"),
            os.path.join(save_dir, sub, f"{sample_id}_neg.{ext}"))


def generate(args, pipeline, dataloader, device, accelerator):
    metadata = []
    output_type = "latent" if args.save_type == "latent" else "pil"

    for batch in tqdm(dataloader, desc="Generating", disable=not accelerator.is_local_main_process):
        ids, tags, prompts, neg_prompts, all_neg = (list(x) for x in batch)

        pos_paths, neg_paths = zip(*(_paths(args.save_dir, args.save_type, i) for i in ids))
        already_done = all(os.path.exists(p) and os.path.exists(n)
                           for p, n in zip(pos_paths, neg_paths))

        if not already_done:
            seeds = [random.randint(0, int(1e9)) for _ in prompts]
            pos_gen = [torch.Generator(device=device).manual_seed(s) for s in seeds]
            neg_gen = [torch.Generator(device=device).manual_seed(s) for s in seeds]
            outs_pos = pipeline(prompts, num_inference_steps=args.num_inference_steps,
                                guidance_scale=args.cfg, generator=pos_gen, output_type=output_type).images
            outs_neg = pipeline(neg_prompts, num_inference_steps=args.num_inference_steps,
                                guidance_scale=args.cfg, generator=neg_gen, output_type=output_type).images
            for pos_path, neg_path, out_pos, out_neg in zip(pos_paths, neg_paths, outs_pos, outs_neg):
                if args.save_type == "latent":
                    save_file({"latent": out_pos}, pos_path)
                    save_file({"latent": out_neg}, neg_path)
                else:
                    out_pos.save(pos_path)
                    out_neg.save(neg_path)

        for sid, tag, prompt, neg_prompt, pos_path, neg_path, negs in zip(
                ids, tags, prompts, neg_prompts, pos_paths, neg_paths, all_neg):
            metadata.append({
                "id": sid, "tag": tag, "prompt": prompt, "neg_prompt": neg_prompt,
                "pos_file": Path(pos_path).name, "neg_file": Path(neg_path).name,
                "all_neg_prompts": negs,
            })
        accelerator.wait_for_everyone()

    all_metadata = accelerator.gather_for_metrics(metadata)
    if accelerator.is_local_main_process:
        flattened = []
        for item in all_metadata:
            flattened.extend(item) if isinstance(item, list) else flattened.append(item)
        with open(args.metadata_file, "w") as f:
            for md in flattened:
                f.write(json.dumps(md) + "\n")


def _load_pipeline(args, device):
    if args.model_type == "sdxl":
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
        scheduler = EulerDiscreteScheduler.from_pretrained(
            args.model_name, subfolder="scheduler", cache_dir=args.cache_dir)
        return StableDiffusionXLPipeline.from_pretrained(
            args.model_name, scheduler=scheduler, torch_dtype=torch.float16,
            variant="fp16", cache_dir=args.cache_dir).to(device)
    if args.model_type == "sd3":
        from diffusers import StableDiffusion3Pipeline
        return StableDiffusion3Pipeline.from_pretrained(
            args.model_name, torch_dtype=torch.float16, cache_dir=args.cache_dir).to(device)
    if args.model_type == "sana":
        from diffusers import SanaPipeline
        dtype = torch.bfloat16 if args.bf16 else torch.float16
        variant = "bf16" if args.bf16 else "fp16"
        pipe = SanaPipeline.from_pretrained(
            args.model_name, torch_dtype=dtype, variant=variant, cache_dir=args.cache_dir).to(device)
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
        return pipe
    from diffusers import StableDiffusionPipeline
    return StableDiffusionPipeline.from_pretrained(
        args.model_name, safety_checker=None, torch_dtype=torch.float16, cache_dir=args.cache_dir).to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_file", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./data/paired_latents/")
    p.add_argument("--model_type", type=str, default="sd15", choices=MODEL_TYPES)
    p.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--num_samples", type=int, default=-1, help="-1 for all.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--save_type", type=str, default="latent", choices=["latent", "image"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--bf16", action="store_true", help="Use bf16 (SANA).")
    args = p.parse_args()
    args.metadata_file = str(Path(args.save_dir) / "metadata.jsonl")
    return args


def main():
    args = parse_args()
    accelerator = Accelerator()
    seed = args.seed + accelerator.process_index
    random.seed(seed)
    torch.manual_seed(seed)
    device = accelerator.device

    sub = "latents" if args.save_type == "latent" else "images"
    os.makedirs(os.path.join(args.save_dir, sub), exist_ok=True)

    dataset = JsonDataset(args.json_file)
    if args.num_samples > 0:
        state = random.getstate()
        random.seed(42)
        indices = random.sample(range(len(dataset)), args.num_samples)
        random.setstate(state)
        dataset = Subset(dataset, indices)

    dataloader = accelerator.prepare(DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate, num_workers=4))

    pipe = _load_pipeline(args, device)
    pipe.set_progress_bar_config(disable=True)
    generate(args, pipe, dataloader, device, accelerator)


if __name__ == "__main__":
    main()
