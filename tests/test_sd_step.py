"""CPU integration smoke test for the SD1.x adapter forward step.

Builds a tiny randomly-initialised UNet (no pretrained download) and exercises
the real per-step tensor plumbing of ``SDAdapter.training_step`` — pair
splitting, the CFG reference target, and the prediction calls — then checks the
shared DPO loss runs and produces a gradient. SDXL / SD3 / SANA forward passes
need real model weights and are exercised on the training cluster.
"""

from types import SimpleNamespace

import torch

from diffusers import DDPMScheduler, UNet2DConditionModel

from ddspo.adapters.sd import SDAdapter
from ddspo.dpo import dpo_loss

CROSS_DIM = 16
SEQ_LEN = 8


def _tiny_unet():
    return UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(8, 16), norm_num_groups=4, cross_attention_dim=CROSS_DIM,
        attention_head_dim=2,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    )


class _FakeTextEncoder(torch.nn.Module):
    def forward(self, input_ids):
        b, length = input_ids.shape
        # Deterministic, differentiable-free stand-in for CLIP hidden states.
        return (torch.ones(b, length, CROSS_DIM),)


def _make_adapter():
    adapter = SDAdapter(sdxl=False)
    adapter.args = SimpleNamespace(guidance_scale=1)
    adapter.noise_scheduler = DDPMScheduler(num_train_timesteps=10)
    adapter.num_train_timesteps = 10
    adapter.text_encoder = _FakeTextEncoder()
    adapter.text_encoder_2 = None
    adapter.vae = None
    adapter.ref_unet = _tiny_unet()
    adapter.pos_lora_unet = None
    adapter.neg_lora_unet = None
    adapter.null_encoder_hidden_states = None
    return adapter


def _batch(b=2):
    return {
        "latents": torch.randn(b, 8, 8, 8),           # (B, 2*C, H, W)
        "timesteps": torch.randint(0, 10, (b,)),
        "paireds": torch.ones(b, dtype=torch.long),
        "pos_input_ids": torch.zeros(b, SEQ_LEN, dtype=torch.long),
        "neg_input_ids": torch.zeros(b, SEQ_LEN, dtype=torch.long),
    }


def test_sd_training_step_and_loss():
    adapter = _make_adapter()
    model = _tiny_unet()
    args = SimpleNamespace(only_cfg=True, lora_path=None, guidance_scale=1, beta_dpo=5000,
                           loss_weighting=None)

    model_pred, ref_pred, target, timesteps = adapter.training_step(
        model, _batch(2), args, torch.float32, torch.device("cpu"))

    assert model_pred.shape == (4, 4, 8, 8)   # (2*B, C, H, W)
    assert ref_pred.shape == model_pred.shape == target.shape
    assert timesteps.shape == (4,)

    loss, metrics = dpo_loss(model_pred, ref_pred, target, args.beta_dpo, timesteps)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
