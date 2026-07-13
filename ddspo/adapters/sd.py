"""Stable Diffusion 1.x / SDXL adapter (UNet, epsilon prediction).

Ported from the original ``ddspo_lora_traget_main.py`` trainer. Supports:

* ``--model_type sd15`` : single CLIP text encoder.
* ``--model_type sdxl``  : dual CLIP text encoders + micro-conditioning
  (``time_ids`` / pooled embeddings).
* ``--lora_path``        : LoRA-target mode. Frozen pre-trained ``pos_lora_unet``
  and ``neg_lora_unet`` supply the preference targets instead of the reference
  UNet's classifier-free-guidance prediction.
"""

import copy
import os

import torch
import torch.utils.data
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, PretrainedConfig

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

from ..data import collate_fn, make_self_training_dataloader
from .base import ModelAdapter


def _import_text_encoder_class(model_path, revision, subfolder="text_encoder"):
    config = PretrainedConfig.from_pretrained(model_path, subfolder=subfolder, revision=revision)
    model_class = config.architectures[0]
    if model_class == "CLIPTextModel":
        return CLIPTextModel
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    raise ValueError(f"{model_class} is not supported.")


class SDAdapter(ModelAdapter):
    """Adapter for SD1.x and SDXL."""

    def __init__(self, sdxl=False):
        self.sdxl = sdxl

    # ---- loading -------------------------------------------------------
    def load(self, args, accelerator):
        self.args = args
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir
        )
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        assert self.noise_scheduler.config.prediction_type == "epsilon"

        if self.sdxl:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer",
                revision=args.revision, use_fast=False, cache_dir=args.cache_dir,
            )
            self.tokenizer_2 = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer_2",
                revision=args.revision, use_fast=False, cache_dir=args.cache_dir,
            )
            cls_one = _import_text_encoder_class(args.pretrained_model_name_or_path, args.revision)
            cls_two = _import_text_encoder_class(
                args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
            self.text_encoder = cls_one.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder",
                revision=args.revision, cache_dir=args.cache_dir)
            self.text_encoder_2 = cls_two.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2",
                revision=args.revision, cache_dir=args.cache_dir)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer",
                revision=args.revision, cache_dir=args.cache_dir)
            self.tokenizer_2 = None
            self.text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder",
                revision=args.revision, cache_dir=args.cache_dir)
            self.text_encoder_2 = None

        self.vae_path = args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path
        self.vae = AutoencoderKL.from_pretrained(
            self.vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision, cache_dir=args.cache_dir)

        self.ref_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            revision=args.revision, cache_dir=args.cache_dir)
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            revision=args.revision, cache_dir=args.cache_dir)

        # Optional frozen pos/neg LoRA UNets for LoRA-target mode.
        self.pos_lora_unet = self.neg_lora_unet = None
        if args.lora_path is not None:
            from peft import PeftModel
            self.pos_lora_unet = PeftModel.from_pretrained(
                copy.deepcopy(self.ref_unet), os.path.join(args.lora_path, "pos_lora_unet"))
            self.neg_lora_unet = PeftModel.from_pretrained(
                copy.deepcopy(self.ref_unet), os.path.join(args.lora_path, "neg_lora_unet"))
            self.pos_lora_unet.requires_grad_(False)
            self.neg_lora_unet.requires_grad_(False)

        # Freeze everything except the trainable UNet.
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)
        self.ref_unet.requires_grad_(False)

        if args.gradient_checkpointing or self.sdxl:
            unet.enable_gradient_checkpointing()
        return unet

    def place_frozen(self, accelerator, weight_dtype):
        device = accelerator.device
        self.vae.to(device, dtype=weight_dtype)
        self.text_encoder.to(device, dtype=weight_dtype)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(device, dtype=weight_dtype)
        self.ref_unet.to(device, dtype=weight_dtype)
        if self.pos_lora_unet is not None:
            self.pos_lora_unet.to(device, dtype=weight_dtype)
            self.neg_lora_unet.to(device, dtype=weight_dtype)

        # Precompute null-prompt embeddings for classifier-free-guidance targets
        # (SD1.x only; SDXL uses the negative-prompt path directly).
        self.null_encoder_hidden_states = None
        if (not self.sdxl) and self.args.guidance_scale != 1:
            null_ids = self.tokenizer(
                [""], padding="max_length", max_length=self.tokenizer.model_max_length,
                return_tensors="pt").input_ids.to(device)
            self.null_encoder_hidden_states = self.text_encoder(null_ids)[0]

    def make_dataloader(self, args):
        return make_self_training_dataloader(
            args, collate_fn(self.tokenizer, self.tokenizer_2), self.num_train_timesteps)

    # ---- prompt encoding ----------------------------------------------
    def _encode_sdxl(self, input_ids_list, device):
        embeds_list = []
        pooled = None
        with torch.no_grad():
            for input_ids, encoder in zip(input_ids_list, [self.text_encoder, self.text_encoder_2]):
                out = encoder(input_ids.to(device), output_hidden_states=True)
                pooled = out[0]
                embeds_list.append(out.hidden_states[-2])
        prompt_embeds = torch.cat(embeds_list, dim=-1)
        bs = prompt_embeds.shape[0]
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled.view(bs, -1)}

    # ---- training step -------------------------------------------------
    def training_step(self, model, batch, args, weight_dtype, device):
        latents = torch.cat(batch["latents"].chunk(2, dim=1)).to(weight_dtype)
        noise = torch.randn_like(latents)
        timesteps = batch["timesteps"].long().to(device).repeat(2)
        # DPO: share timestep and noise across each preference pair.
        noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # --- conditioning ---
        if self.sdxl:
            add_time_ids = torch.tensor(
                [args.resolution, args.resolution, 0, 0, args.resolution, args.resolution],
                dtype=weight_dtype, device=device)[None, :].repeat(timesteps.size(0), 1)
            prompt_batch = self._encode_sdxl([batch["pos_input_ids"], batch["pos_input_ids_2"]], device)
            neg_prompt_batch = self._encode_sdxl([batch["neg_input_ids"], batch["neg_input_ids_2"]], device)
            prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
            prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
            added_cond_kwargs = {"time_ids": add_time_ids,
                                 "text_embeds": prompt_batch["pooled_prompt_embeds"]}
            cond = prompt_batch["prompt_embeds"]
        else:
            enc = self.text_encoder(batch["pos_input_ids"].to(device))[0]
            pos_encoder_hidden_states = enc
            cond = enc.repeat(2, 1, 1)
            added_cond_kwargs = None

        # --- target ---
        target = noise.clone()
        paireds = torch.zeros_like(batch["paireds"]) if args.only_cfg else batch["paireds"]
        stdpo_indices = (paireds == 0).nonzero(as_tuple=True)[0]

        if len(stdpo_indices) > 0:
            pos_indices = stdpo_indices
            neg_indices = stdpo_indices + paireds.shape[0]
            final_indices = torch.cat([pos_indices, neg_indices], dim=0)
            stdpo_latents = noisy_latents[final_indices]
            stdpo_timesteps = timesteps[final_indices]

            if args.lora_path is not None:
                self._lora_target(target, batch, device, pos_indices, neg_indices, noisy_latents,
                                  timesteps, cond, added_cond_kwargs, prompt_batch if self.sdxl else None)
            else:
                self._cfg_target(target, batch, device, stdpo_indices, final_indices, stdpo_latents,
                                 stdpo_timesteps, added_cond_kwargs,
                                 prompt_batch if self.sdxl else None,
                                 neg_prompt_batch if self.sdxl else None,
                                 pos_encoder_hidden_states if not self.sdxl else None,
                                 add_time_ids if self.sdxl else None, args)

        # --- predictions ---
        model_pred = model(noisy_latents, timesteps, cond, added_cond_kwargs=added_cond_kwargs).sample
        with torch.no_grad():
            ref_pred = self.ref_unet(
                noisy_latents, timesteps, cond, added_cond_kwargs=added_cond_kwargs).sample.detach()
        return model_pred, ref_pred, target, timesteps

    def _lora_target(self, target, batch, device, pos_indices, neg_indices, noisy_latents,
                     timesteps, cond, added_cond_kwargs, prompt_batch):
        pos_noisy, pos_ts = noisy_latents[pos_indices], timesteps[pos_indices]
        neg_noisy, neg_ts = noisy_latents[neg_indices], timesteps[neg_indices]
        with torch.no_grad():
            if self.sdxl:
                cond_prompt = prompt_batch["prompt_embeds"].chunk(2)[0][pos_indices]
                cond_added = {"time_ids": added_cond_kwargs["time_ids"][pos_indices],
                              "text_embeds": prompt_batch["pooled_prompt_embeds"].chunk(2)[0][pos_indices]}
                pos_pred = self.pos_lora_unet(pos_noisy, pos_ts, cond_prompt, added_cond_kwargs=cond_added).sample
                neg_pred = self.neg_lora_unet(neg_noisy, neg_ts, cond_prompt, added_cond_kwargs=cond_added).sample
            else:
                cond_prompt = cond.chunk(2)[0][pos_indices]
                pos_pred = self.pos_lora_unet(pos_noisy, pos_ts, cond_prompt).sample
                neg_pred = self.neg_lora_unet(neg_noisy, neg_ts, cond_prompt).sample
        target[pos_indices] = pos_pred.to(target.dtype)
        target[neg_indices] = neg_pred.to(target.dtype)

    def _cfg_target(self, target, batch, device, stdpo_indices, final_indices, stdpo_latents,
                    stdpo_timesteps, added_cond_kwargs, prompt_batch, neg_prompt_batch,
                    pos_encoder_hidden_states, add_time_ids, args):
        if self.sdxl:
            direction_embeds = torch.cat(
                [prompt_batch["prompt_embeds"].chunk(2)[0][stdpo_indices],
                 neg_prompt_batch["prompt_embeds"][stdpo_indices]], dim=0)
            direction_added = {
                "time_ids": add_time_ids[final_indices],
                "text_embeds": torch.cat(
                    [prompt_batch["pooled_prompt_embeds"].chunk(2)[0][stdpo_indices],
                     neg_prompt_batch["pooled_prompt_embeds"][stdpo_indices]], dim=0)}
        else:
            stdpo_pos = pos_encoder_hidden_states[stdpo_indices]
            stdpo_neg = self.text_encoder(batch["neg_input_ids"][stdpo_indices].to(device))[0]
            direction_embeds = torch.cat([stdpo_pos, stdpo_neg], dim=0)
            direction_added = None

        with torch.no_grad():
            ref_pos_neg = self.ref_unet(
                stdpo_latents, stdpo_timesteps, direction_embeds,
                added_cond_kwargs=direction_added).sample.detach()
            if args.guidance_scale != 1 and not self.sdxl:
                null_embeds = self.null_encoder_hidden_states.repeat(len(stdpo_indices) * 2, 1, 1)
                ref_null = self.ref_unet(
                    stdpo_latents, stdpo_timesteps, null_embeds,
                    added_cond_kwargs=direction_added).sample.detach()
                cfg_target = ref_null + args.guidance_scale * (ref_pos_neg - ref_null)
            else:
                cfg_target = ref_pos_neg
        target[final_indices] = cfg_target.to(target.dtype)

    # ---- save ----------------------------------------------------------
    def save(self, args, accelerator, model):
        unet = accelerator.unwrap_model(model)
        if self.sdxl:
            vae = AutoencoderKL.from_pretrained(
                self.vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision)
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path, unet=unet, vae=vae, revision=args.revision)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path, text_encoder=self.text_encoder,
                vae=self.vae, unet=unet, revision=args.revision, cache_dir=args.cache_dir)
        pipeline.save_pretrained(args.output_dir)
