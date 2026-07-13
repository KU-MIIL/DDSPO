"""Model adapters and the factory that maps ``--model_type`` to an adapter.

The concrete adapters are imported lazily inside :func:`get_adapter` so that
training one family does not require the diffusers classes of the others (e.g.
SANA's ``AutoencoderDC``) to be importable.
"""

# Gemma2 prompt-enhancement instruction used by SANA. Kept here (free of heavy
# imports) so ``ddspo.args`` can reference it as a default.
DEFAULT_COMPLEX_HUMAN_INSTRUCTION = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt:",
]


def get_adapter(model_type):
    """Return an (unloaded) adapter instance for ``model_type``."""
    if model_type == "sd15":
        from .sd import SDAdapter
        return SDAdapter(sdxl=False)
    if model_type == "sdxl":
        from .sd import SDAdapter
        return SDAdapter(sdxl=True)
    if model_type == "sd3":
        from .sd3 import SD3Adapter
        return SD3Adapter()
    if model_type == "sana":
        from .sana import SANAAdapter
        return SANAAdapter()
    raise ValueError(f"Unknown model_type: {model_type}")


__all__ = ["get_adapter", "DEFAULT_COMPLEX_HUMAN_INSTRUCTION"]
