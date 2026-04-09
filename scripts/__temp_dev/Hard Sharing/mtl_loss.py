# mtl_loss.py
# ─────────────────────────────────────────────────────────────────────────────
# Loss functions for MTL training.
#
# UncertaintyWeightedMTLLoss  (main — Kendall et al. 2018)
#   - Learns one log_σ² per trait
#   - Automatically downweights hard/noisy traits
#   - Per-trait masked MSE respects NaN labels and sPlot/GBIF source weights
#
# PerTraitMaskedLoss  (utility used by both train and evaluate loops)
#   - Returns (31,) tensor of per-trait losses without uncertainty weighting
#   - Used for per-trait r computation and logging
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── per-trait masked weighted MSE ───────────────────────────────────────────


def per_trait_masked_loss(
    y_pred: torch.Tensor,  # (B, T)  or  (B, T, H, W)
    y_true: torch.Tensor,  # same shape
    source_mask: torch.Tensor,  # same shape, int64: 0=missing, 1=GBIF, 2=sPlot
    w_gbif: float = 1.0,
    w_splot: float = 2.0,
    reduction: str = "mean",  # "mean" | "sum"
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-trait (column-wise) weighted masked MSE.

    Works on both (B, T) and (B, T, H, W) tensors — flattens to (B*H*W, T)
    internally so the same function serves both the STL dense path and the
    MTL center-pixel path.

    Returns:
        losses : (T,) per-trait mean squared error, NaN where no valid pixels
    """
    # Flatten spatial dims if present
    if y_pred.ndim == 4:
        B, T, H, W = y_pred.shape
        y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, T)
        y_true = y_true.permute(0, 2, 3, 1).reshape(-1, T)
        source_mask = source_mask.permute(0, 2, 3, 1).reshape(-1, T)

    # y_pred, y_true, source_mask : (N, T)
    n_traits = y_pred.shape[1]

    # Weight map: 0=missing, w_gbif=GBIF, w_splot=sPlot
    weights = torch.zeros_like(y_true)
    weights = torch.where(source_mask == 1, torch.full_like(weights, w_gbif), weights)
    weights = torch.where(source_mask == 2, torch.full_like(weights, w_splot), weights)

    # Validity: finite predictions, finite targets, non-zero weight
    valid = torch.isfinite(y_pred) & torch.isfinite(y_true) & (weights > 0)

    se = (y_pred - y_true).pow(2)

    trait_losses = torch.full((n_traits,), float("nan"), device=y_pred.device)
    for t in range(n_traits):
        v = valid[:, t]
        if v.any():
            w_t = weights[v, t]
            se_t = se[v, t]
            if reduction == "mean":
                trait_losses[t] = (se_t * w_t).sum() / w_t.sum().clamp_min(eps)
            else:
                trait_losses[t] = (se_t * w_t).sum()

    return trait_losses  # (T,)


# ─── uncertainty-weighted MTL loss ───────────────────────────────────────────


class UncertaintyWeightedMTLLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting (Kendall et al., NeurIPS 2018).

    For each trait i:
        weighted_loss_i = exp(-log_σ_i²) * L_i  +  log_σ_i²

    Total loss = sum over all traits with valid observations.

    The log_σ_i² parameters are learned alongside the model weights.
    After training, inspect them: high σ_i → hard/noisy trait (low r in STL),
    low σ_i → well-conditioned trait (high r in STL).

    Args:
        n_traits   : number of output traits
        w_gbif     : weight for GBIF-sourced pixels
        w_splot    : weight for sPlot-sourced pixels (recommended: 2× GBIF)
        init_log_var: initial value of log_σ² (0.0 = equal weights at start)

    Usage in train loop:
        loss_fn = UncertaintyWeightedMTLLoss(n_traits=31)
        # Add its parameters to the optimizer:
        optimizer = Adam([
            {"params": model.parameters()},
            {"params": loss_fn.parameters(), "lr": 1e-3},
        ])
    """

    def __init__(
        self,
        n_traits: int = 31,
        w_gbif: float = 1.0,
        w_splot: float = 2.0,
        init_log_var: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_traits = n_traits
        self.w_gbif = w_gbif
        self.w_splot = w_splot

        # One learnable log(σ²) per trait
        self.log_var = nn.Parameter(torch.full((n_traits,), init_log_var))

    def forward(
        self,
        y_pred: torch.Tensor,  # (B, T) or (B, T, H, W)
        y_true: torch.Tensor,  # same shape
        source_mask: torch.Tensor,  # same shape, int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss  : scalar loss for backward()
            trait_losses: (T,) per-trait raw MSE (before weighting) for logging
        """
        # Per-trait masked MSE: (T,)
        trait_losses = per_trait_masked_loss(
            y_pred,
            y_true,
            source_mask,
            w_gbif=self.w_gbif,
            w_splot=self.w_splot,
        )

        # Uncertainty weighting:  exp(-log_σ²) * L_i + log_σ²
        # Only sum over traits that have valid observations (not NaN)
        valid_traits = torch.isfinite(trait_losses)

        if not valid_traits.any():
            return y_pred.sum() * 0.0, trait_losses

        precision = torch.exp(-self.log_var)  # (T,)
        weighted = precision * trait_losses  # (T,)
        reg = self.log_var  # (T,)  regularizer

        total_loss = (weighted[valid_traits] + reg[valid_traits]).sum()

        return total_loss, trait_losses

    def get_task_weights(self) -> torch.Tensor:
        """
        Returns the current effective weights (1/σ²) per trait.
        Use for logging and analysis.
        """
        return torch.exp(-self.log_var).detach()

    def get_log_var(self) -> torch.Tensor:
        """Returns log_σ² values per trait (learned)."""
        return self.log_var.detach()


# ─── sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 16, 31

    y_pred = torch.randn(B, T)
    y_true = torch.randn(B, T)
    src_mask = torch.ones(B, T, dtype=torch.int64)
    # Simulate some missing labels
    src_mask[:, 15:] = 0
    src_mask[0, :5] = 2  # sPlot

    loss_fn = UncertaintyWeightedMTLLoss(n_traits=T)
    total, trait_losses = loss_fn(y_pred, y_true, src_mask)

    print(f"Total loss  : {total.item():.4f}")
    print(f"Trait losses: {trait_losses[:5].tolist()}")
    print(f"log_var     : {loss_fn.get_log_var()[:5].tolist()}")
    print(f"task weights: {loss_fn.get_task_weights()[:5].tolist()}")
