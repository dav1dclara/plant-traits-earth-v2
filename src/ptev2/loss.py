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
