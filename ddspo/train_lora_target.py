#!/usr/bin/env python
# coding=utf-8
"""Pre-train the winning / losing model pair for DDSPO DD-CPP (data-driven).

This trains two LoRA adapters on a shared base UNet:

* ``pos_lora_unet`` on the positive (preferred) latents;
* ``neg_lora_unet`` on the negative (degraded) latents;

both with a plain epsilon-MSE objective and the positive prompt as conditioning.
The resulting ``pos_lora_unet/`` and ``neg_lora_unet/`` directories are passed to
``ddspo/train.py`` via ``--lora_path`` to supply the contrastive-policy-pair
targets. Supports SD1.x and SDXL. This is the DD-CPP instantiation: the pair is
trained on an existing preference dataset (pos = chosen, neg = rejected).
"""

import argparse
import copy
import logging
import math
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from .adapters.sd import _import_text_encoder_class
from .data import collate_fn, make_self_training_dataloader

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    p = argparse.ArgumentParser(description="DDSPO LoRA-target (pos/neg) pre-training.")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--train_data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="ddspo-lora-target")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--sdxl", action="store_true")
    p.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--tracker_project_name", type=str, default="ddspo-lora-target")
    # LoRA
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=4.0)
    p.add_argument("--lora_lr", type=float, default=1e-4)
    # dataset knobs (kept off by default for the MSE pre-training)
    p.add_argument("--rand_cond", action="store_true")
    p.add_argument("--rand_cond_lambda", type=float, default=20)
    p.add_argument("--rand_cond_pt", type=str, default="sigmoid")
    p.add_argument("--extra_text_path", type=str, nargs="+", default=None)
    p.add_argument("--timestep_sampling", type=str, default=None)

    args = p.parse_args()
    if args.resolution is None:
        args.resolution = 1024 if args.sdxl else 512
    return args


def _encode_sdxl(text_encoders, input_ids_list, device):
    embeds_list, pooled = [], None
    with torch.no_grad():
        for input_ids, encoder in zip(input_ids_list, text_encoders):
            out = encoder(input_ids.to(device), output_hidden_states=True)
            pooled = out[0]
            embeds_list.append(out.hidden_states[-2])
    prompt_embeds = torch.cat(embeds_list, dim=-1)
    return prompt_embeds, pooled.view(prompt_embeds.shape[0], -1)


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision, log_with=args.report_to,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, args.logging_dir)))
    logging.basicConfig(level=logging.INFO)
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir)
    assert noise_scheduler.config.prediction_type == "epsilon"

    # tokenizers / text encoders
    if args.sdxl:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,
            use_fast=False, cache_dir=args.cache_dir)
        tokenizer_2 = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision,
            use_fast=False, cache_dir=args.cache_dir)
        cls_one = _import_text_encoder_class(args.pretrained_model_name_or_path, args.revision)
        cls_two = _import_text_encoder_class(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
        text_encoder = cls_one.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision,
            cache_dir=args.cache_dir)
        text_encoder_2 = cls_two.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision,
            cache_dir=args.cache_dir)
        text_encoders = [text_encoder, text_encoder_2]
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,
            cache_dir=args.cache_dir)
        tokenizer_2 = None
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision,
            cache_dir=args.cache_dir)
        text_encoders = [text_encoder]

    base_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
        cache_dir=args.cache_dir)

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.0)
    pos_lora_unet = get_peft_model(copy.deepcopy(base_unet), lora_config)
    neg_lora_unet = get_peft_model(copy.deepcopy(base_unet), lora_config)
    del base_unet
    for lora in (pos_lora_unet, neg_lora_unet):
        for name, param in lora.named_parameters():
            param.requires_grad = "lora_" in name
        if args.gradient_checkpointing:
            lora.enable_gradient_checkpointing()

    for te in text_encoders:
        te.requires_grad_(False)

    lora_params = ([p for p in pos_lora_unet.parameters() if p.requires_grad]
                   + [p for p in neg_lora_unet.parameters() if p.requires_grad])
    optimizer = torch.optim.AdamW(lora_params, lr=args.lora_lr)

    train_dataloader = make_self_training_dataloader(
        args, collate_fn(tokenizer, tokenizer_2), noise_scheduler.config.num_train_timesteps)

    overrode = args.max_train_steps is None
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes)

    pos_lora_unet, neg_lora_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pos_lora_unet, neg_lora_unet, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        accelerator.mixed_precision, torch.float32)
    for te in text_encoders:
        te.to(accelerator.device, dtype=weight_dtype)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {})

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process,
                        desc="LoRA-target")
    for _ in range(args.num_train_epochs):
        pos_lora_unet.train()
        neg_lora_unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(pos_lora_unet):
                latents = torch.cat(batch["latents"].chunk(2, dim=1)).to(weight_dtype)
                bsz = latents.shape[0] // 2
                pos_latents, neg_latents = latents[:bsz], latents[bsz:]
                noise_pos = torch.randn_like(pos_latents)
                noise_neg = torch.randn_like(neg_latents)
                timesteps = batch["timesteps"].long().to(latents.device)
                noisy_pos = noise_scheduler.add_noise(pos_latents, noise_pos, timesteps)
                noisy_neg = noise_scheduler.add_noise(neg_latents, noise_neg, timesteps)

                if args.sdxl:
                    add_time_ids = torch.tensor(
                        [args.resolution, args.resolution, 0, 0, args.resolution, args.resolution],
                        dtype=weight_dtype, device=accelerator.device)[None, :].repeat(bsz, 1)
                    cond, cond_pooled = _encode_sdxl(
                        text_encoders, [batch["pos_input_ids"], batch["pos_input_ids_2"]],
                        accelerator.device)
                    cond_added = {"time_ids": add_time_ids, "text_embeds": cond_pooled}
                    pos_pred = pos_lora_unet(noisy_pos, timesteps, cond, added_cond_kwargs=cond_added).sample
                    neg_pred = neg_lora_unet(noisy_neg, timesteps, cond, added_cond_kwargs=cond_added).sample
                else:
                    with torch.no_grad():
                        cond = text_encoder(batch["pos_input_ids"].to(accelerator.device))[0]
                    pos_pred = pos_lora_unet(noisy_pos, timesteps, cond).sample
                    neg_pred = neg_lora_unet(noisy_neg, timesteps, cond).sample

                pos_loss = F.mse_loss(pos_pred.float(), noise_pos.float())
                neg_loss = F.mse_loss(neg_pred.float(), noise_neg.float())
                loss = pos_loss + neg_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"lora_loss": loss.detach().item(),
                                 "pos_loss": pos_loss.detach().item(),
                                 "neg_loss": neg_loss.detach().item()}, step=global_step)
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
            progress_bar.set_postfix(loss=loss.detach().item())
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(pos_lora_unet).save_pretrained(os.path.join(args.output_dir, "pos_lora_unet"))
        accelerator.unwrap_model(neg_lora_unet).save_pretrained(os.path.join(args.output_dir, "neg_lora_unet"))
        logger.info(f"Saved pos/neg LoRA UNets to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
