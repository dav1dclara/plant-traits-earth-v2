"""
Loss functions for MTL.
Provides masked dense losses for single-task and multi-task training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """Mean squared error over finite values only."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(prediction) & torch.isfinite(target)
        if not valid.any():
            return torch.tensor(0.0, device=prediction.device)
        return F.mse_loss(prediction[valid], target[valid], reduction=self.reduction)


class MTLLoss(nn.Module):
    """Weighted multi-task loss for dense predictions."""

    def __init__(
        self, task_weights: list[float] | None = None, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.task_weights = task_weights or [1.0]
        self.reduction = reduction

    def forward(
        self, predictions: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for idx, (pred, target) in enumerate(zip(predictions, targets)):
            valid = torch.isfinite(pred) & torch.isfinite(target)
            if valid.any():
                losses.append(
                    F.mse_loss(pred[valid], target[valid], reduction=self.reduction)
                )
            else:
                losses.append(torch.tensor(0.0, device=pred.device))

        if len(self.task_weights) != len(losses):
            weights = [1.0] * len(losses)
        else:
            weights = self.task_weights
        return sum(w * loss for w, loss in zip(weights, losses))


class WeightedMaskedDenseLoss(nn.Module):
    """Dense regression loss restricted to mask areas."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = torch.isfinite(prediction) & torch.isfinite(target) & (source_mask > 0)
        if not valid.any():
            return torch.tensor(0.0, device=prediction.device)
        return F.mse_loss(prediction[valid], target[valid], reduction=self.reduction)
