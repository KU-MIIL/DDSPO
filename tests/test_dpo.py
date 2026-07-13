"""CPU smoke tests for the shared Diffusion-DPO loss."""

import math

import torch

from ddspo.dpo import dpo_loss


def _tensors(b=2, c=4, h=8, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (2 * b, c, h, w)
    model_pred = torch.randn(shape, generator=g, requires_grad=True)
    ref_pred = torch.randn(shape, generator=g)
    target = torch.randn(shape, generator=g)
    timesteps = torch.randint(0, 1000, (2 * b,))
    return model_pred, ref_pred, target, timesteps


def test_loss_is_finite_and_has_metrics():
    model_pred, ref_pred, target, timesteps = _tensors()
    loss, metrics = dpo_loss(model_pred, ref_pred, target, beta_dpo=5000, timesteps=timesteps)
    assert torch.isfinite(loss)
    assert set(metrics) == {"raw_model_loss", "raw_ref_loss", "implicit_acc"}
    assert 0.0 <= metrics["implicit_acc"].item() <= 1.0


def test_equal_predictions_give_log2_loss():
    # When model == ref, model_diff == ref_diff -> inside_term == 0 -> loss == log 2.
    _, _, target, timesteps = _tensors()
    pred = torch.randn_like(target)
    loss, metrics = dpo_loss(pred.clone().requires_grad_(True), pred, target,
                             beta_dpo=5000, timesteps=timesteps)
    assert abs(loss.item() - math.log(2)) < 1e-4
    assert metrics["implicit_acc"].item() == 0.0


def test_gradient_flows_to_model_pred():
    model_pred, ref_pred, target, timesteps = _tensors()
    loss, _ = dpo_loss(model_pred, ref_pred, target, beta_dpo=5000, timesteps=timesteps)
    loss.backward()
    assert model_pred.grad is not None and torch.isfinite(model_pred.grad).all()


def test_linear_weighting_runs():
    model_pred, ref_pred, target, timesteps = _tensors()
    loss, _ = dpo_loss(model_pred, ref_pred, target, beta_dpo=5000, timesteps=timesteps,
                       loss_weighting="linear", num_train_timesteps=1000)
    assert torch.isfinite(loss)
