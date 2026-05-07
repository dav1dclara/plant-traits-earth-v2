"""Compatibility module for legacy multitask imports.

Model implementations now live in dedicated modules:
- ptev2.models.pixel_mlp
- ptev2.models.stl
- ptev2.models.mtl
- ptev2.models.mmoe

This file keeps the historical loss utilities and re-exports the model
classes so older checkpoints and imports still work.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptev2.models.mmoe import GatedMMoEModel, GatedMMoEModelV3
from ptev2.models.mtl import MTLModel
from ptev2.models.pixel_mlp import PixelMLP
from ptev2.models.stl import STLModel


def _per_trait_masked_loss_v2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
    w_gbif: float = 1.0,
    w_splot: float = 2.0,
    error_type: str = "mse",
    huber_delta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if prediction.ndim == 4:
        prediction = prediction.permute(0, 2, 3, 1).reshape(-1, prediction.shape[1])
        target = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
        source_mask = source_mask.permute(0, 2, 3, 1).reshape(-1, source_mask.shape[1])

    weights = torch.zeros_like(target, dtype=prediction.dtype)
    weights = torch.where(source_mask == 1, torch.full_like(weights, w_gbif), weights)
    weights = torch.where(source_mask == 2, torch.full_like(weights, w_splot), weights)

    valid = torch.isfinite(prediction) & torch.isfinite(target) & (weights > 0)
    valid_f = valid.to(dtype=prediction.dtype)

    pred_safe = torch.where(valid, prediction, torch.zeros_like(prediction))
    tgt_safe = torch.where(valid, target, torch.zeros_like(target))

    if error_type == "mse":
        error = (pred_safe - tgt_safe).pow(2)
    elif error_type in ("smooth_l1", "huber"):
        error = F.smooth_l1_loss(
            pred_safe, tgt_safe, reduction="none", beta=huber_delta
        )
    else:
        raise ValueError(f"Unsupported error_type: {error_type}")

    numerator = (error * weights).sum(dim=0)
    denominator = (weights * valid_f).sum(dim=0).clamp_min(eps)
    trait_losses = numerator / denominator

    no_valid = valid_f.sum(dim=0) == 0
    if bool(no_valid.any()):
        trait_losses = trait_losses.clone()
        trait_losses[no_valid] = float("nan")
    return trait_losses


class UncertaintyWeightedMTLLossV2(nn.Module):
    def __init__(
        self,
        n_traits: int = 37,
        w_gbif: float = 1.0,
        w_splot: float = 2.0,
        init_log_sigma_sq: float = 0.0,
        error_type: str = "smooth_l1",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.log_sigma_sq = nn.Parameter(
            torch.full((n_traits,), float(init_log_sigma_sq), dtype=torch.float32)
        )
        self.w_gbif = float(w_gbif)
        self.w_splot = float(w_splot)
        self.error_type = str(error_type)
        self.huber_delta = float(huber_delta)

    def loss_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trait_losses = _per_trait_masked_loss_v2(
            prediction,
            target,
            source_mask,
            w_gbif=self.w_gbif,
            w_splot=self.w_splot,
            error_type=self.error_type,
            huber_delta=self.huber_delta,
        )

        valid = torch.isfinite(trait_losses)
        if not bool(valid.any()):
            zero = prediction.sum() * 0.0
            return zero, zero

        trait_losses = torch.nan_to_num(trait_losses, nan=0.0)
        weighted = (
            torch.exp(-self.log_sigma_sq[valid]) * trait_losses[valid]
            + self.log_sigma_sq[valid]
        )
        numerator = weighted.sum()
        denominator = valid.sum().to(dtype=prediction.dtype)
        return numerator, denominator

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        numerator, denominator = self.loss_components(prediction, target, source_mask)
        if not bool((denominator > 0).item()):
            return prediction.sum() * 0.0
        return numerator / denominator


class GradNormMTLLoss(nn.Module):
    def __init__(
        self,
        n_traits: int = 37,
        alpha: float = 1.5,
        ema_decay: float = 0.98,
        gradnorm_weight: float = 0.1,
        w_gbif: float = 1.0,
        w_splot: float = 16.0,
        error_type: str = "smooth_l1",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_traits = int(n_traits)
        self.alpha = float(alpha)
        self.ema_decay = float(ema_decay)
        self.gradnorm_weight = float(gradnorm_weight)
        self.w_gbif = float(w_gbif)
        self.w_splot = float(w_splot)
        self.error_type = str(error_type)
        self.huber_delta = float(huber_delta)

        self.log_weights = nn.Parameter(torch.zeros(self.n_traits))
        self.register_buffer("ema_losses", torch.ones(self.n_traits))
        self.register_buffer("initial_losses", torch.zeros(self.n_traits))
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def weights(self) -> torch.Tensor:
        w = self.log_weights.exp()
        return w / w.sum() * self.n_traits

    def _update_ema(self, per_task_losses: torch.Tensor) -> None:
        with torch.no_grad():
            valid = torch.isfinite(per_task_losses)
            losses = (
                per_task_losses.detach()
                .where(valid, torch.ones_like(per_task_losses))
                .clamp_min(1e-8)
            )
            if not bool(self.initialized.item()):
                self.initial_losses.copy_(losses)
                self.ema_losses.copy_(losses)
                self.initialized.fill_(True)
            else:
                self.ema_losses = torch.where(
                    valid,
                    self.ema_decay * self.ema_losses + (1 - self.ema_decay) * losses,
                    self.ema_losses,
                )

    def _target_weights(self) -> torch.Tensor:
        if not bool(self.initialized.item()):
            return torch.ones(self.n_traits, device=self.log_weights.device)
        L_hat = self.ema_losses / (self.initial_losses + 1e-8)
        r = L_hat / (L_hat.mean() + 1e-8)
        target = r.pow(self.alpha)
        return target / target.sum() * self.n_traits

    def loss_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        per_task_losses = _per_trait_masked_loss_v2(
            prediction,
            target,
            source_mask,
            w_gbif=self.w_gbif,
            w_splot=self.w_splot,
            error_type=self.error_type,
            huber_delta=self.huber_delta,
        )

        self._update_ema(per_task_losses)

        valid = torch.isfinite(per_task_losses)
        if not bool(valid.any()):
            zero = prediction.sum() * 0.0
            return zero, zero

        weights = self.weights
        weighted_losses = weights[valid] * per_task_losses[valid]

        target_w = self._target_weights().detach()
        gnorm_reg = ((weights - target_w) ** 2).mean()

        numerator = weighted_losses.sum() + self.gradnorm_weight * gnorm_reg
        denominator = valid.sum().to(dtype=prediction.dtype)
        return numerator, denominator

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        num, den = self.loss_components(prediction, target, source_mask)
        if not bool((den > 0).item()):
            return prediction.sum() * 0.0
        return num / den


class AuxQuantileHead(nn.Module):
    def __init__(self, n_traits: int) -> None:
        super().__init__()
        self.n_traits = int(n_traits)
        self.head = nn.Conv2d(
            n_traits, n_traits * 2, kernel_size=1, groups=n_traits, bias=True
        )
        with torch.no_grad():
            self.head.weight.fill_(1.0)
            self.head.bias[:n_traits].fill_(-1.0)
            self.head.bias[n_traits:].fill_(1.0)

    def forward(self, mean_pred: torch.Tensor) -> torch.Tensor:
        return self.head(mean_pred)
