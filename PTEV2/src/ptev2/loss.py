import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Configurable MSE loss for Hydra instantiation."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """MSE over finite values only (ignores NaN/Inf in prediction/target)."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(prediction) & torch.isfinite(target)
        if not bool(valid.any()):
            return prediction.sum() * 0.0
        return F.mse_loss(
            prediction[valid],
            target[valid],
            reduction=self.reduction,
        )


def per_trait_masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
    w_gbif: float = 1.0,
    w_splot: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a per-output masked regression loss over dense maps."""
    if prediction.shape != target.shape or source_mask.shape != target.shape:
        raise ValueError(
            "prediction, target, and source_mask must have identical shapes, got "
            f"{prediction.shape}, {target.shape}, and {source_mask.shape}"
        )

    if prediction.ndim == 4:
        batch_size, n_outputs, height, width = prediction.shape
        prediction = prediction.permute(0, 2, 3, 1).reshape(-1, n_outputs)
        target = target.permute(0, 2, 3, 1).reshape(-1, n_outputs)
        source_mask = source_mask.permute(0, 2, 3, 1).reshape(-1, n_outputs)

    weights = torch.zeros_like(target, dtype=prediction.dtype)
    weights = torch.where(source_mask == 1, torch.full_like(weights, w_gbif), weights)
    weights = torch.where(source_mask == 2, torch.full_like(weights, w_splot), weights)

    valid = torch.isfinite(prediction) & torch.isfinite(target) & (weights > 0)
    valid_f = valid.to(dtype=prediction.dtype)

    pred_safe = torch.where(valid, prediction, torch.zeros_like(prediction))
    target_safe = torch.where(valid, target, torch.zeros_like(target))
    squared_error = (pred_safe - target_safe).pow(2)

    if reduction == "mean":
        numerator = (squared_error * weights).sum(dim=0)
        denominator = (weights * valid_f).sum(dim=0).clamp_min(eps)
        trait_losses = numerator / denominator
    elif reduction == "sum":
        trait_losses = (squared_error * weights * valid_f).sum(dim=0)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    no_valid = valid_f.sum(dim=0) == 0
    if bool(no_valid.any()):
        trait_losses = trait_losses.clone()
        trait_losses[no_valid] = float("nan")

    return trait_losses


class UncertaintyWeightedMTLLoss(nn.Module):
    """Per-output uncertainty-weighted dense regression loss for MTL/MMoE runs."""

    def __init__(
        self,
        n_traits: int = 37,
        w_gbif: float = 1.0,
        w_splot: float = 2.0,
        init_log_sigma_sq: float = 0.0,
    ) -> None:
        super().__init__()
        self.log_sigma_sq = nn.Parameter(
            torch.full((n_traits,), float(init_log_sigma_sq), dtype=torch.float32)
        )
        self.w_gbif = float(w_gbif)
        self.w_splot = float(w_splot)

    def loss_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trait_losses = per_trait_masked_loss(
            prediction,
            target,
            source_mask,
            w_gbif=self.w_gbif,
            w_splot=self.w_splot,
            reduction="mean",
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


class WeightedMaskedDenseLoss(nn.Module):
    """Weighted masked dense regression loss for tensors shaped (B, C, H, W)."""

    def __init__(
        self,
        error_type: str = "mse",
        huber_delta: float = 1.0,
        w_gbif: float = 1.0,
        w_splot: float = 2.0,
    ) -> None:
        super().__init__()
        self.error_type = str(error_type).lower()
        self.huber_delta = float(huber_delta)
        self.w_gbif = float(w_gbif)
        self.w_splot = float(w_splot)

    def _error_map(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.error_type == "mse":
            return (prediction - target).pow(2)
        if self.error_type in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
                beta=self.huber_delta,
            )
        raise ValueError(
            "Unsupported error_type for WeightedMaskedDenseLoss: "
            f"{self.error_type}. Use one of ['mse', 'smooth_l1', 'huber']."
        )

    def loss_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes must match, got {prediction.shape} vs {target.shape}"
            )
        if source_mask.shape != target.shape:
            raise ValueError(
                "source_mask and target shapes must match, got "
                f"{source_mask.shape} vs {target.shape}"
            )
        source_mask_i = source_mask.to(torch.int64)
        valid = (
            torch.isfinite(prediction) & torch.isfinite(target) & (source_mask_i > 0)
        )

        weights = torch.zeros_like(target, dtype=prediction.dtype)
        weights = torch.where(
            source_mask_i == 1,
            torch.full_like(weights, self.w_gbif),
            weights,
        )
        weights = torch.where(
            source_mask_i == 2,
            torch.full_like(weights, self.w_splot),
            weights,
        )

        weighted_valid = valid.to(dtype=prediction.dtype) * weights
        # Idea 3: optional per-position quality weight (e.g. splot_count / splot_std).
        # Multiplies both numerator and denominator, so it re-weights the contribution
        # of each chip/trait relative to the batch average.
        if sample_weight is not None:
            weighted_valid = weighted_valid * sample_weight
        denominator = weighted_valid.sum()
        if not bool((denominator > 0).item()):
            zero = prediction.sum() * 0.0
            return zero, denominator

        # Important: avoid propagating NaN gradients from invalid target locations.
        # We sanitize both tensors before error computation and re-apply the valid mask.
        pred_safe = torch.where(valid, prediction, torch.zeros_like(prediction))
        target_safe = torch.where(valid, target, torch.zeros_like(target))
        error_map = self._error_map(pred_safe, target_safe)
        numerator = (error_map * weighted_valid).sum()
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
