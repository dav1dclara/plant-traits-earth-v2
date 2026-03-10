# ptev2/models/traitPatchCNN.py (FCN-style ResNet patch CNN for center-pixel regression)
# ResNet-style (4-block) patch CNN for center-pixel regression at 22 km (and transferable to other resolutions).
#
# Based on the patch-based protocol from Sialelli et al. (2025) and Padarian et al. (2019),
# adapted for multi-task plant trait prediction.
#
# Key properties (matching PlantTraits.Earth v2 specs):
# - Patch-based learning: input is (B, C, k, k), k=15 or 25
# - Optional lat/lon trig encodings are handled by simply increasing in_channels (+4) in the DATA pipeline
# - Encoder: 4 conv blocks, 3x3 conv + Norm + ReLU; strided conv in every other block (blocks 2 and 4 by default)
# - Head: 1x1 conv producing 31 outputs; we take ONLY the center pixel (center-pixel regression)
# - Lightweight residual connections where shapes match
#
# This is intentionally simple/stable and avoids any dense decoder, since global context
# is already provided by the input patch, and center-pixel supervision is substantially cheaper.

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(
    norm: Literal["bn", "gn", "none"],
    num_channels: int,
    gn_groups: int = 8,
) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        # Ensure groups divide channels
        g = min(gn_groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm='{norm}'")


class ConvBlock(nn.Module):
    """
    A single conv block:
      Conv2d(3x3, stride) -> Norm -> ReLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm = _make_norm(norm, out_ch, gn_groups=gn_groups)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResPatchCenterCNN(nn.Module):
    """
    Minimal ResNet-style encoder (4 conv blocks) + 1x1 head,
    predicting only the CENTER pixel trait vector.

    This model provides a baseline for patch-based multi-task regression,
    leveraging spatial context while maintaining computational efficiency.

    Args:
        in_channels: Number of input channels (e.g., 50 predictors, or 54 with trig encodings)
        n_traits: Number of output traits (default: 31)
        base_channels: Base number of channels in first conv block (default: 32)
        norm: Normalization type - 'bn' (BatchNorm), 'gn' (GroupNorm), or 'none'
        gn_groups: Number of groups for GroupNorm (default: 8)
        stride_blocks: Stride pattern for 4 blocks (default: (1, 2, 1, 2) for downsampling in blocks 2 and 4)
        dropout_p: Dropout probability (default: 0.0)
        use_residual: Whether to add residual connections where shapes match (default: True)

    Input:  x shape (B, C, k, k)  where k is odd (e.g., 15 or 25 recommended)
    Output: y shape (B, n_traits)
    """

    def __init__(
        self,
        in_channels: int,
        n_traits: int = 31,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        # Stride pattern "every other block":
        # block2 and block4 downsample by 2 -> total downsample factor 4
        stride_blocks: tuple[int, int, int, int] = (1, 2, 1, 2),
        dropout_p: float = 0.0,
        # If True, add lightweight residual connections where shapes match
        use_residual: bool = True,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # 4 conv blocks (small ResNet-style CNN)
        # Channel schedule: C -> c1 -> c1 -> c2 -> c3 (simple widening with depth)
        self.b1 = ConvBlock(
            in_channels,
            c1,
            stride=stride_blocks[0],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )
        self.b2 = ConvBlock(
            c1,
            c1,
            stride=stride_blocks[1],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )
        self.b3 = ConvBlock(
            c1,
            c2,
            stride=stride_blocks[2],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )
        self.b4 = ConvBlock(
            c2,
            c3,
            stride=stride_blocks[3],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )

        self.use_residual = use_residual

        # 1x1 conv head produces per-pixel trait logits; we will take only center pixel.
        self.head = nn.Conv2d(c3, n_traits, kernel_size=1, bias=True)

    @staticmethod
    def _center_index(h: int, w: int) -> tuple[int, int]:
        return h // 2, w // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, k, k)
        returns: (B, n_traits)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to be (B,C,H,W), got {tuple(x.shape)}")
        _, _, H, W = x.shape
        if (H % 2) == 0 or (W % 2) == 0:
            raise ValueError(
                f"Patch size should be odd (e.g., 15 or 25). Got H={H}, W={W}"
            )

        # Encoder
        z1 = self.b1(x)

        z2 = self.b2(z1)
        if self.use_residual and z2.shape == z1.shape:
            z2 = z2 + z1  # tiny ResNet-style skip when shape matches

        z3 = self.b3(z2)

        z4 = self.b4(z3)
        if self.use_residual and z4.shape == z3.shape:
            z4 = z4 + z3

        # Head (per-pixel), then pick center pixel only
        y_map = self.head(z4)  # (B, n_traits, H', W')
        _, _, Hp, Wp = y_map.shape
        ci, cj = self._center_index(Hp, Wp)

        y_center = y_map[:, :, ci, cj]  # (B, n_traits)
        return y_center


# ----------------------------
# Optional: masked loss (keep in training.py if you prefer)
# ----------------------------
def masked_mse_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    y_pred, y_true: (B, T)
    mask: (B, T), 1=observed, 0=missing
    """
    if mask is None:
        return F.mse_loss(y_pred, y_true)
    mask = mask.float()
    se = (y_pred - y_true) ** 2
    se = se * mask
    return se.sum() / mask.sum().clamp_min(eps)


if __name__ == "__main__":
    # quick sanity check
    torch.manual_seed(0)
    model = ResPatchCenterCNN(
        in_channels=54, n_traits=31, base_channels=32, norm="gn", dropout_p=0.1
    )
    x = torch.randn(8, 54, 15, 15)  # 50 predictors + 4 trig coord channels = 54
    y = model(x)
    print(y.shape)  # torch.Size([8, 31])
