"""Stable Diffusion 3 adapter (MMDiT, flow matching, LoRA).

Ported from the original ``ddspo_sd3_lora.py`` trainer. SD3 is fine-tuned with a
LoRA adapter on the transformer; the reference prediction is obtained by running
the same transformer with the adapter temporarily disabled (``lora_off``). Three
text encoders (2x CLIP + T5) produce the joint prompt embedding.
"""

import copy

import torch
import torch.utils.data
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.training_utils import compute_density_for_timestep_sampling

from ..data import collate_fn, make_self_training_dataloader
from ._lora import lora_off
from .base import ModelAdapter

DEFAULT_LORA_TARGETS = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]


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

        transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        for te in self.text_encoders:
            te.requires_grad_(False)

        # Reference forward = same transformer with LoRA disabled (no separate copy).
        from peft import LoraConfig
        target_modules = ([m.strip() for m in args.lora_layers.split(",")]
                          if args.lora_layers else DEFAULT_LORA_TARGETS)
        transformer.add_adapter(LoraConfig(
            r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
            target_modules=target_modules))
        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
        return transformer

    def place_frozen(self, accelerator, weight_dtype):
        device = accelerator.device
        self.vae.to(device, dtype=torch.float32)  # VAE kept fp32 for stability
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

        def run(hidden, ehs, pooled_proj, ts):
            return model(hidden_states=hidden, timestep=ts, encoder_hidden_states=ehs,
                         pooled_projections=pooled_proj, return_dict=False)[0]

        def precondition(pred, noisy):
            return pred * (-sigmas) + noisy if args.precondition_outputs else pred

        dup_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
        dup_pooled = torch.cat([pooled, pooled], dim=0)
        model_pred = precondition(run(noisy_model_input, dup_embeds, dup_pooled, timesteps), noisy_model_input)
        with torch.no_grad(), lora_off(model):
            ref_pred = precondition(run(noisy_model_input, dup_embeds, dup_pooled, timesteps), noisy_model_input)

        target = (model_input if args.precondition_outputs else (noise - model_input)).clone()

        paireds = torch.zeros_like(batch["paireds"]) if args.cpp else batch["paireds"]
        cpp_indices = (paireds == 0).nonzero(as_tuple=True)[0]
        if len(cpp_indices) > 0:
            pos_indices = cpp_indices
            neg_indices = cpp_indices + paireds.shape[0]
            final_indices = torch.cat([pos_indices, neg_indices], dim=0)
            cpp_noisy = noisy_model_input[final_indices]
            cpp_ts = timesteps[final_indices]
            cpp_embeds = torch.cat([prompt_embeds[pos_indices], neg_prompt_embeds[pos_indices]], dim=0)
            cpp_pooled = torch.cat([pooled[pos_indices], neg_pooled[pos_indices]], dim=0)
            with torch.no_grad(), lora_off(model):
                ref_pair = run(cpp_noisy, cpp_embeds, cpp_pooled, cpp_ts)
                if args.precondition_outputs:
                    ref_pair = ref_pair * (-sigmas[final_indices]) + cpp_noisy
            target[final_indices] = ref_pair.to(dtype=target.dtype)

        return model_pred, ref_pred, target, timesteps

    def save(self, args, accelerator, model):
        from peft.utils import get_peft_model_state_dict
        transformer = accelerator.unwrap_model(model)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=args.output_dir, transformer_lora_layers=transformer_lora_layers)
