from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceAwareLoss(nn.Module):
    """Source-aware regression loss for the active pipeline modes.

    Supported modes:
    - splot_only
    - source_aware_comb_single_head
    """

    def __init__(
        self,
        mode: str = "splot_only",
        primary_dataset: str = "splot",
        auxiliary_dataset: str = "gbif",
        lambda_gbif: float = 0.1,
        error_type: str = "smooth_l1",
        huber_delta: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.primary_dataset = str(primary_dataset)
        self.auxiliary_dataset = str(auxiliary_dataset)
        self.lambda_gbif = float(lambda_gbif)
        self.error_type = str(error_type).lower()
        self.huber_delta = float(huber_delta)
        self.eps = float(eps)

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
            f"Unsupported error_type '{self.error_type}'. Use mse|smooth_l1|huber."
        )

    def _dataset_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes must match, got {prediction.shape} vs {target.shape}"
            )
        if prediction.ndim != 4:
            raise ValueError(f"Expected dense maps (B,C,H,W), got {prediction.shape}.")

        valid = torch.isfinite(prediction) & torch.isfinite(target)
        pred_safe = torch.where(valid, prediction, torch.zeros_like(prediction))
        target_safe = torch.where(valid, target, torch.zeros_like(target))
        err = self._error_map(pred_safe, target_safe)

        valid_f = valid.to(dtype=prediction.dtype)
        num_per_trait = (err * valid_f).sum(dim=(0, 2, 3))
        den_per_trait = valid_f.sum(dim=(0, 2, 3))
        has_valid = den_per_trait > 0

        if not bool(has_valid.any()):
            zero = prediction.sum() * 0.0
            return zero, zero, zero

        trait_loss = num_per_trait[has_valid] / den_per_trait[has_valid].clamp_min(
            self.eps
        )
        numerator = trait_loss.sum()
        denominator = has_valid.sum().to(dtype=prediction.dtype)
        valid_count = den_per_trait[has_valid].sum()
        return numerator, denominator, valid_count

    def loss_components(
        self,
        prediction: torch.Tensor,
        target_bundle: dict[str, dict[str, torch.Tensor]],
        _unused_source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        y_splot = target_bundle.get(self.primary_dataset, {}).get("y")
        y_gbif = target_bundle.get(self.auxiliary_dataset, {}).get("y")

        if self.mode == "splot_only":
            if y_splot is None:
                raise ValueError("splot_only requires primary dataset targets.")
            num_splot, den_splot, valid_splot = self._dataset_components(
                prediction, y_splot
            )
            loss_splot = (
                num_splot / den_splot
                if bool((den_splot > 0).item())
                else prediction.sum() * 0.0
            )
            details = {
                "loss_total": float(loss_splot.detach().item()),
                "loss_splot": float(loss_splot.detach().item()),
                "loss_gbif": float("nan"),
                "valid_splot": float(valid_splot.detach().item()),
                "valid_gbif": 0.0,
                "lambda_gbif": self.lambda_gbif,
                "effective_gbif_splot_ratio": float("nan"),
                "valid_total": float(valid_splot.detach().item()),
            }
            return num_splot, den_splot, details

        if self.mode != "source_aware_comb_single_head":
            raise ValueError(
                f"Unsupported mode '{self.mode}'. Use splot_only|source_aware_comb_single_head."
            )
        if y_splot is None or y_gbif is None:
            raise ValueError(
                "source_aware_comb_single_head requires both primary and auxiliary targets."
            )

        num_splot, den_splot, valid_splot = self._dataset_components(
            prediction, y_splot
        )
        num_gbif, den_gbif, valid_gbif = self._dataset_components(prediction, y_gbif)

        loss_splot = (
            num_splot / den_splot
            if bool((den_splot > 0).item())
            else prediction.sum() * 0.0
        )
        loss_gbif = (
            num_gbif / den_gbif
            if bool((den_gbif > 0).item())
            else prediction.sum() * 0.0
        )
        combined_num = num_splot + self.lambda_gbif * num_gbif
        combined_den = den_splot + self.lambda_gbif * den_gbif
        total_loss = (
            combined_num / combined_den
            if bool((combined_den > 0).item())
            else prediction.sum() * 0.0
        )

        details = {
            "loss_total": float(total_loss.detach().item()),
            "loss_splot": float(loss_splot.detach().item()),
            "loss_gbif": float(loss_gbif.detach().item()),
            "valid_splot": float(valid_splot.detach().item()),
            "valid_gbif": float(valid_gbif.detach().item()),
            "lambda_gbif": self.lambda_gbif,
            "effective_gbif_splot_ratio": float(
                (self.lambda_gbif * valid_gbif / valid_splot).detach().item()
            )
            if bool((valid_splot > 0).item())
            else float("nan"),
            "valid_total": float((valid_splot + valid_gbif).detach().item()),
        }
        return combined_num, combined_den, details

    def forward(
        self,
        prediction: torch.Tensor,
        target_bundle: dict[str, dict[str, torch.Tensor]],
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numerator, denominator, _ = self.loss_components(
            prediction, target_bundle, source_mask
        )
        if bool((denominator > 0).item()):
            return numerator / denominator
        return prediction.sum() * 0.0
