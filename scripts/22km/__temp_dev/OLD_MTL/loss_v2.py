"""
Loss functions for MTL training.
Includes uncertainty-weighted loss and per-trait masked losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Per-trait masked loss ───────────────────────────────────────────────────
def per_trait_masked_loss(
    y_pred: torch.Tensor,  # (B, T) or (B, T, H, W)
    y_true: torch.Tensor,  # same shape
    source_mask: torch.Tensor,  # same shape, int64: 0=missing, 1=GBIF, 2=sPlot
    w_gbif: float = 1.0,
    w_splot: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-trait weighted masked MSE.

    Args:
        y_pred: Predictions (B, T) or (B, T, H, W)
        y_true: Ground truth (same shape)
        source_mask: Source indicator (0=missing, 1=GBIF, 2=sPlot)
        w_gbif: Weight for GBIF data
        w_splot: Weight for sPlot data
        reduction: "mean" or "sum"

    Returns:
        losses: (T,) per-trait MSE, NaN where no valid pixels
    """
    # Flatten spatial dims if present
    if y_pred.ndim == 4:
        B, T, H, W = y_pred.shape
        y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, T)
        y_true = y_true.permute(0, 2, 3, 1).reshape(-1, T)
        source_mask = source_mask.permute(0, 2, 3, 1).reshape(-1, T)

    # (N, T) tensors
    n_traits = y_pred.shape[1]

    # Compute weights: 0=missing, w_gbif=GBIF, w_splot=sPlot
    weights = torch.zeros_like(y_true, dtype=y_pred.dtype)
    weights = torch.where(source_mask == 1, torch.full_like(weights, w_gbif), weights)
    weights = torch.where(source_mask == 2, torch.full_like(weights, w_splot), weights)

    # Validity: finite predictions, finite targets, non-zero weight
    valid = torch.isfinite(y_pred) & torch.isfinite(y_true) & (weights > 0)
    valid_f = valid.to(dtype=y_pred.dtype)

    # Safe squared error
    y_true_safe = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    se = (y_pred - y_true_safe).pow(2)
    se = se * valid_f

    if reduction == "mean":
        numerator = (se * weights).sum(dim=0)  # (T,)
        denominator = (weights * valid_f).sum(dim=0).clamp_min(eps)  # (T,)
        trait_losses = numerator / denominator
    else:
        trait_losses = (se * weights * valid_f).sum(dim=0)  # (T,)

    # Traits with zero valid pixels remain NaN
    no_valid = valid_f.sum(dim=0) == 0
    if no_valid.any():
        trait_losses = trait_losses.clone()
        trait_losses[no_valid] = float("nan")

    return trait_losses  # (T,)


# ─── Uncertainty-Weighted MTL Loss ───────────────────────────────────────────
class UncertaintyWeightedMTLLoss(nn.Module):
    """
    Homoscedastic multi-task learning loss with learned uncertainty weights.

    For each trait i:
        loss_i = exp(-log_σ_i²) * MSE_i  +  log_σ_i²

    The log_σ_i² parameters are learnable and automatically calibrate
    the relative importance of each trait based on their difficulty.

    Higher σ_i → trait i is harder, so its loss is downweighted.
    This allows automatic balancing of multi-task losses.
    """

    def __init__(
        self,
        n_traits: int = 37,
        w_gbif: float = 1.0,
        w_splot: float = 2.0,
        init_log_sigma_sq: float = 0.0,
    ):
        super().__init__()
        # Learnable log(σ²) for each trait
        self.log_sigma_sq = nn.Parameter(
            torch.full((n_traits,), init_log_sigma_sq, dtype=torch.float32)
        )
        self.n_traits = n_traits
        self.w_gbif = w_gbif
        self.w_splot = w_splot

    def forward(
        self,
        y_pred: torch.Tensor,  # (B, T, H, W)
        y_true: torch.Tensor,  # (B, T, H, W)
        source_mask: torch.Tensor,  # (B, T, H, W), int64
    ) -> torch.Tensor:
        """
        Compute uncertainty-weighted loss.

        Returns: scalar loss
        """
        # Per-trait masked MSE
        trait_mse = per_trait_masked_loss(
            y_pred,
            y_true,
            source_mask,
            w_gbif=self.w_gbif,
            w_splot=self.w_splot,
            reduction="mean",
        )  # (T,)

        # Replace NaN with 0 (no valid pixels → no loss contribution)
        trait_mse = torch.nan_to_num(trait_mse, nan=0.0)

        # Uncertainty-weighted loss
        # loss = Σ_i [ exp(-log_σ_i²) * MSE_i  +  log_σ_i² ]
        weighted_loss = torch.exp(-self.log_sigma_sq) * trait_mse + self.log_sigma_sq

        # Return mean loss across traits
        return weighted_loss.mean()


# ─── Simple Masked MSE Loss ──────────────────────────────────────────────────
class MaskedMSELoss(nn.Module):
    """
    Simple masked MSE loss over finite values only.
    Used as fallback if uncertainty weighting is not needed.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(prediction) & torch.isfinite(target)
        if not valid.any():
            return torch.tensor(0.0, device=prediction.device)
        return F.mse_loss(prediction[valid], target[valid], reduction=self.reduction)
