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
