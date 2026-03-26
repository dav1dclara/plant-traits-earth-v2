from typing import Literal

import torch
import torch.nn as nn


class PadarianInspiredPatchCNN(nn.Module):
    """
    Padarian-inspired CNN baseline adapted for dense spatial regression.

    Input:
        (B, C_in, H, W)

    Output:
        (B, out_channels, H, W)

    Notes:
        - Keeps a shallow, lightweight architecture.
        - Uses fully convolutional heads so it can be trained with dense targets.
        - `n_traits` is kept as backward-compatible alias for `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 31,
        n_traits: int | None = None,
        hidden_dim: int = 64,
        dropout_p: float = 0.3,
        head_type: Literal["single", "multi"] = "single",
    ):
        super().__init__()
        if n_traits is not None:
            out_channels = int(n_traits)

        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}.")

        self.out_channels = out_channels
        self.n_traits = out_channels
        self.head_type = head_type

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.PReLU(num_parameters=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(16, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.PReLU(num_parameters=hidden_dim),
            nn.Dropout(p=dropout_p),
        )

        if head_type == "single":
            self.head = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
            self.heads = None
        elif head_type == "multi":
            self.head = None
            self.heads = nn.ModuleList(
                [
                    nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
                    for _ in range(out_channels)
                ]
            )
        else:
            raise ValueError(
                f"Unknown head_type='{head_type}'. Use 'single' or 'multi'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x to be (B,C,H,W), got {tuple(x.shape)}")

        h = self.backbone(x)  # (B, hidden_dim, H/2, W/2)
        h = nn.functional.interpolate(
            h,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # (B, hidden_dim, H, W)

        if self.head_type == "single":
            return self.head(h)  # (B, out_channels, H, W)

        return torch.cat(
            [head(h) for head in self.heads], dim=1
        )  # (B, out_channels, H, W)
