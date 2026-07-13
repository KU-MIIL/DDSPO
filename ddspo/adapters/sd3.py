"""Stable Diffusion 3 adapter (MMDiT, flow matching).

Ported from the original ``ddspo_sd3.py`` trainer. SD3 full-finetunes the
transformer and keeps a separate frozen reference transformer for the DPO term.
Three text encoders (2x CLIP + T5) produce the joint prompt embedding.
"""

import copy

import torch
import torch.utils.data
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from ..data import collate_fn, make_self_training_dataloader
from .base import ModelAdapter


def _encode_with_clip(text_encoder, tokenizer, prompt, device):
    text_input_ids = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    out = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled = out[0]
    embeds = out.hidden_states[-2].to(dtype=text_encoder.dtype, device=device)
    return embeds, pooled


def _encode_with_t5(text_encoder, tokenizer, prompt, device, max_sequence_length):
    text_input_ids = tokenizer(
        prompt, padding="max_length", max_length=max_sequence_length, truncation=True,
        add_special_tokens=True, return_tensors="pt").input_ids
    return text_encoder(text_input_ids.to(device))[0].to(dtype=text_encoder.dtype, device=device)


class SD3Adapter(ModelAdapter):
    def load(self, args, accelerator):
        self.args = args
        p, rev, var = args.pretrained_model_name_or_path, args.revision, args.variant
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(p, subfolder="scheduler")
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps

        self.tokenizers = [
            CLIPTokenizer.from_pretrained(p, subfolder="tokenizer", revision=rev),
            CLIPTokenizer.from_pretrained(p, subfolder="tokenizer_2", revision=rev),
            T5TokenizerFast.from_pretrained(p, subfolder="tokenizer_3", revision=rev),
        ]
        self.text_encoders = [
            CLIPTextModelWithProjection.from_pretrained(p, subfolder="text_encoder", revision=rev, variant=var),
            CLIPTextModelWithProjection.from_pretrained(p, subfolder="text_encoder_2", revision=rev, variant=var),
            T5EncoderModel.from_pretrained(p, subfolder="text_encoder_3", revision=rev, variant=var),
        ]
        self.vae = AutoencoderKL.from_pretrained(p, subfolder="vae", revision=rev, variant=var)
        transformer = SD3Transformer2DModel.from_pretrained(p, subfolder="transformer", revision=rev, variant=var)
        self.ref_transformer = SD3Transformer2DModel.from_pretrained(
            p, subfolder="transformer", revision=rev, variant=var)

        self.vae.requires_grad_(False)
        self.ref_transformer.requires_grad_(False)
        for te in self.text_encoders:
            te.requires_grad_(False)
        transformer.requires_grad_(True)
        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
        return transformer

    def place_frozen(self, accelerator, weight_dtype):
        device = accelerator.device
        # VAE is always kept in fp32 for numerical stability; the trainable
        # transformer is left in fp32 too (handled by accelerator autocast).
        self.vae.to(device, dtype=torch.float32)
        self.ref_transformer.to(device, dtype=weight_dtype)
        for te in self.text_encoders:
            te.to(device, dtype=weight_dtype)

    def make_dataloader(self, args):
        # collate_fn() with no tokenizer -> raw prompt strings (encoded in-loop).
        return make_self_training_dataloader(args, collate_fn(), self.num_train_timesteps)

    def _encode_prompt(self, prompt, device):
        clip_embeds, clip_pooled = [], []
        for te, tok in zip(self.text_encoders[:2], self.tokenizers[:2]):
            embeds, pooled = _encode_with_clip(te, tok, prompt, device)
            clip_embeds.append(embeds)
            clip_pooled.append(pooled)
        clip_prompt_embeds = torch.cat(clip_embeds, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled, dim=-1)
        t5_embed = _encode_with_t5(self.text_encoders[-1], self.tokenizers[-1], prompt, device,
                                   self.args.max_sequence_length)
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_embed.shape[-1] - clip_prompt_embeds.shape[-1]))
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_embed], dim=-2)
        return prompt_embeds.to(device), pooled_prompt_embeds.to(device)

    def _get_sigmas(self, timesteps, device, n_dim, dtype):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps.to(device)]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def training_step(self, model, batch, args, weight_dtype, device):
        model_input = torch.cat(batch["latents"].chunk(2, dim=1)).to(dtype=weight_dtype)
        noise = torch.randn_like(model_input)
        noise = noise if args.pos_neg_different_noise else noise.chunk(2)[0].repeat(2, 1, 1, 1)
        bsz = model_input.shape[0]

        u_half = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme, batch_size=bsz // 2,
            logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale)
        u = torch.cat([u_half, u_half], dim=0)
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        sigmas = self._get_sigmas(timesteps, device, model_input.ndim, model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        prompt_embeds, pooled = self._encode_prompt(batch["pos_prompts"], device)
        neg_prompt_embeds, neg_pooled = self._encode_prompt(batch["neg_prompts"], device)

        def run(transformer, hidden, ehs, pooled_proj, ts):
            return transformer(hidden_states=hidden, timestep=ts, encoder_hidden_states=ehs,
                               pooled_projections=pooled_proj, return_dict=False)[0]

        dup_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
        dup_pooled = torch.cat([pooled, pooled], dim=0)
        model_pred = run(model, noisy_model_input, dup_embeds, dup_pooled, timesteps)
        if args.precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_model_input

        target = model_input if args.precondition_outputs else (noise - model_input)
        target = target.clone()

        paireds = torch.zeros_like(batch["paireds"]) if args.only_cfg else batch["paireds"]
        stdpo_indices = (paireds == 0).nonzero(as_tuple=True)[0]
        if len(stdpo_indices) > 0:
            pos_indices = stdpo_indices
            neg_indices = stdpo_indices + paireds.shape[0]
            final_indices = torch.cat([pos_indices, neg_indices], dim=0)
            stdpo_noisy = noisy_model_input[final_indices]
            stdpo_ts = timesteps[final_indices]
            stdpo_embeds = torch.cat([prompt_embeds[pos_indices], neg_prompt_embeds[pos_indices]], dim=0)
            stdpo_pooled = torch.cat([pooled[pos_indices], neg_pooled[pos_indices]], dim=0)
            with torch.no_grad():
                ref_pos_neg = run(self.ref_transformer, stdpo_noisy, stdpo_embeds, stdpo_pooled, stdpo_ts)
                if args.precondition_outputs:
                    ref_pos_neg = ref_pos_neg * (-sigmas[final_indices]) + stdpo_noisy
            target[final_indices] = ref_pos_neg.to(dtype=target.dtype)

        with torch.no_grad():
            ref_pred = run(self.ref_transformer, noisy_model_input, dup_embeds, dup_pooled, timesteps)
            if args.precondition_outputs:
                ref_pred = ref_pred * (-sigmas) + noisy_model_input

        return model_pred, ref_pred, target, timesteps

    def save(self, args, accelerator, model):
        transformer = accelerator.unwrap_model(model)
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.pretrained_model_name_or_path, transformer=transformer)
        pipeline.save_pretrained(args.output_dir)
