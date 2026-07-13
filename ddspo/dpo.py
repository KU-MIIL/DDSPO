"""Diffusion-DPO loss shared by every model family.

The loss math is identical across SD1.5 / SDXL / SD3 / SANA; only the way the
predictions and targets are produced differs (handled by the model adapters).
The first half of the batch is the preferred sample (``y_w``), the second half
the unpreferred one (``y_l``).
"""

import torch
import torch.nn.functional as F


def dpo_loss(model_pred, ref_pred, target, beta_dpo, timesteps=None,
             loss_weighting=None, num_train_timesteps=1000):
    """Compute the Diffusion-DPO loss.

    Args:
        model_pred: Prediction of the trainable model, shape (2*B, ...).
        ref_pred: Prediction of the frozen reference model, shape (2*B, ...).
        target: Regression target, shape (2*B, ...).
        beta_dpo: DPO temperature controlling the strength of the KL penalty.
        timesteps: Per-sample timesteps, shape (2*B,). Required for weighting.
        loss_weighting: None or "linear" (weight by t / (T-1)).
        num_train_timesteps: T, used for the "linear" weighting schedule.

    Returns:
        (loss, metrics) where metrics is a dict with raw MSE terms and the
        implicit accuracy used for logging.
    """
    per_dim = list(range(1, model_pred.ndim))

    model_losses = (model_pred.float() - target.float()).pow(2).mean(dim=per_dim)
    model_losses_w, model_losses_l = model_losses.chunk(2)
    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
    model_diff = model_losses_w - model_losses_l

    ref_losses = (ref_pred.float() - target.float()).pow(2).mean(dim=per_dim)
    ref_losses_w, ref_losses_l = ref_losses.chunk(2)
    raw_ref_loss = ref_losses.mean()
    ref_diff = ref_losses_w - ref_losses_l

    inside_term = -0.5 * beta_dpo * (model_diff - ref_diff)
    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)

    if loss_weighting == "linear":
        assert timesteps is not None, "linear loss weighting needs timesteps"
        t = timesteps.chunk(2)[0].float().to(inside_term.device)
        weights = t / (num_train_timesteps - 1)
        loss = (-F.logsigmoid(inside_term) * weights).mean()
    else:
        loss = -F.logsigmoid(inside_term).mean()

    metrics = {
        "raw_model_loss": raw_model_loss,
        "raw_ref_loss": raw_ref_loss,
        "implicit_acc": implicit_acc,
    }
    return loss, metrics
