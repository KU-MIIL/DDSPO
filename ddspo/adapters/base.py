"""Model-adapter interface.

Each supported model family (SD1.5/SDXL, SD3, SANA) implements this interface.
The shared training loop in ``ddspo/train.py`` owns everything that is identical
across families (accelerate setup, optimizer, checkpointing, logging and the
Diffusion-DPO loss); the adapter owns everything model-specific (loading, prompt
encoding, the forward diffusion noising, the prediction call and the CFG /
LoRA-target construction).

The key method is :meth:`training_step`, which is a faithful port of the
per-step body of the original per-model trainer. It returns predictions and a
target already expressed in the model's prediction space, so the shared loop can
apply the same DPO loss to all families.
"""

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Interface implemented by every model family."""

    #: Number of diffusion training timesteps (set in :meth:`load`).
    num_train_timesteps = 1000

    @abstractmethod
    def load(self, args, accelerator):
        """Load scheduler, tokenizers, text encoders, VAE and the two models.

        Stores the frozen modules on ``self`` and returns the trainable model
        (UNet / transformer) so the caller can build the optimizer and call
        ``accelerator.prepare`` on it.
        """

    @abstractmethod
    def place_frozen(self, accelerator, weight_dtype):
        """Move frozen modules to the accelerator device / dtype.

        Also precompute anything reused every step (e.g. null-prompt embeddings).
        """

    @abstractmethod
    def make_dataloader(self, args):
        """Build and return the training ``DataLoader``."""

    @abstractmethod
    def training_step(self, model, batch, args, weight_dtype, device):
        """Run the model-specific forward pass for one batch.

        Returns:
            (model_pred, ref_pred, target, timesteps) with the predictions and
            target in the model's prediction space and ``timesteps`` shaped
            (2*B,) for optional loss weighting.
        """

    @abstractmethod
    def save(self, args, accelerator, model):
        """Serialize the trained pipeline to ``args.output_dir``."""
