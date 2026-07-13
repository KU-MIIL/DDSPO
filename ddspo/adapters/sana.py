"""SANA adapter (SanaTransformer, flow matching, DC-AE, Gemma2).

Ported from the original ``dpo_cfg_randcond_sana_re.py`` trainer. SANA trains a
LoRA adapter on the transformer; the reference prediction is obtained by running
the same transformer with the adapter temporarily disabled (``lora_off``).
Prompts are encoded by Gemma2 through ``SanaPipeline.encode_prompt``.
"""

import copy
from contextlib import contextmanager

import torch
import torch.utils.data
from transformers import AutoTokenizer, Gemma2Model

from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler, SanaPipeline, SanaTransformer2DModel
from diffusers.training_utils import compute_density_for_timestep_sampling

from ..data import collate_fn, make_self_training_dataloader
from .base import ModelAdapter


def _lora_layers(model):
    from peft.tuners.lora import LoraLayer
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def _lora_enable(model, flag=True):
    for m in _lora_layers(model):
        m.enable_adapters(flag)


@contextmanager
def lora_off(model):
    _lora_enable(model, False)
    try:
        yield
    finally:
        _lora_enable(model, True)


class SANAAdapter(ModelAdapter):
    def load(self, args, accelerator):
        self.args = args
        p, rev, var = args.pretrained_model_name_or_path, args.revision, args.variant
        self.tokenizer = AutoTokenizer.from_pretrained(p, subfolder="tokenizer", revision=rev)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(p, subfolder="scheduler", revision=rev)
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps

        self.text_encoder = Gemma2Model.from_pretrained(p, subfolder="text_encoder", revision=rev, variant=var)
        self.vae = AutoencoderDC.from_pretrained(p, subfolder="vae", revision=rev, variant=var)
        transformer = SanaTransformer2DModel.from_pretrained(p, subfolder="transformer", revision=rev, variant=var)

        transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        from peft import LoraConfig
        target_modules = ([m.strip() for m in args.lora_layers.split(",")]
                          if args.lora_layers else ["to_k", "to_q", "to_v"])
        transformer.add_adapter(LoraConfig(
            r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=target_modules))
        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
        return transformer

    def place_frozen(self, accelerator, weight_dtype):
        device = accelerator.device
        # SANA-specific dtype rules: VAE always fp32, Gemma2 always bf16.
        self.vae.to(device, dtype=torch.float32)
        self.text_encoder.to(device, dtype=torch.bfloat16)
        self.text_encoding_pipeline = SanaPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path, vae=None, transformer=None,
            text_encoder=self.text_encoder, tokenizer=self.tokenizer).to(device)

    def make_dataloader(self, args):
        # collate_fn() with no tokenizer -> raw prompt strings (encoded in-loop).
        return make_self_training_dataloader(args, collate_fn(), self.num_train_timesteps)

    def _encode_prompt(self, prompt, transformer_dtype):
        with torch.no_grad():
            embeds, mask, _, _ = self.text_encoding_pipeline.encode_prompt(
                prompt, max_sequence_length=self.args.max_sequence_length,
                complex_human_instruction=self.args.complex_human_instruction)
        return embeds.to(transformer_dtype), mask

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

        dtype = weight_dtype
        prompt_embeds, prompt_mask = self._encode_prompt(batch["pos_prompts"], dtype)
        neg_prompt_embeds, neg_prompt_mask = self._encode_prompt(batch["neg_prompts"], dtype)

        dup_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
        dup_mask = torch.cat([prompt_mask, prompt_mask], dim=0)

        def run(hidden, ehs, mask, ts):
            return model(hidden_states=hidden, encoder_hidden_states=ehs,
                         encoder_attention_mask=mask, timestep=ts, return_dict=False)[0]

        model_pred = run(noisy_model_input, dup_embeds, dup_mask, timesteps)
        with torch.no_grad(), lora_off(model):
            ref_pred = run(noisy_model_input, dup_embeds, dup_mask, timesteps)

        target = (noise - model_input).clone()

        paireds = torch.zeros_like(batch["paireds"]) if args.cpp else batch["paireds"]
        cpp_indices = (paireds == 0).nonzero(as_tuple=True)[0]
        if len(cpp_indices) > 0:
            pos_indices = cpp_indices
            neg_indices = cpp_indices + paireds.shape[0]
            final_indices = torch.cat([pos_indices, neg_indices], dim=0)
            cpp_noisy = noisy_model_input[final_indices]
            cpp_ts = timesteps[final_indices]
            cpp_embeds = torch.cat([prompt_embeds[cpp_indices], neg_prompt_embeds[cpp_indices]], dim=0)
            cpp_mask = torch.cat([prompt_mask[cpp_indices], neg_prompt_mask[cpp_indices]], dim=0)
            with torch.no_grad(), lora_off(model):
                ref_pos_neg = run(cpp_noisy, cpp_embeds, cpp_mask, cpp_ts)
            target[final_indices] = ref_pos_neg.to(dtype=target.dtype)

        return model_pred, ref_pred, target, timesteps

    def save(self, args, accelerator, model):
        from peft.utils import get_peft_model_state_dict
        transformer = accelerator.unwrap_model(model)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        SanaPipeline.save_lora_weights(
            save_directory=args.output_dir, transformer_lora_layers=transformer_lora_layers)
