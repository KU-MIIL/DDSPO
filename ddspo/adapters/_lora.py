"""Small PEFT helpers shared by the LoRA-based adapters (SD3, SANA).

Kept free of diffusers imports so it can be used by any adapter. The reference
(frozen) prediction for LoRA training is obtained by running the *same*
transformer with its adapter temporarily disabled.
"""

from contextlib import contextmanager


def _lora_layers(model):
    from peft.tuners.lora import LoraLayer
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def lora_enable(model, flag=True):
    for m in _lora_layers(model):
        m.enable_adapters(flag)


@contextmanager
def lora_off(model):
    """Temporarily disable all LoRA adapters (for the reference forward pass)."""
    lora_enable(model, False)
    try:
        yield
    finally:
        lora_enable(model, True)
