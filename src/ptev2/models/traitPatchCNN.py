from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def _make_norm(
    norm: Literal["bn", "gn", "none"],
    num_channels: int,
    gn_groups: int = 8,
) -> nn.Module:
    """Create normalization layer."""
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)

    if norm == "gn":
        # Ensure the number of groups divides num_channels
        g = min(gn_groups, num_channels)
        while g > 1 and num_channels % g != 0:
            g -= 1
        return nn.GroupNorm(g, num_channels)

    if norm == "none":
        return nn.Identity()

    raise ValueError(f"Unknown norm='{norm}'")


class ResidualBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions.

    Main path:
        Conv3x3 -> Norm -> ReLU -> Dropout -> Conv3x3 -> Norm

    Skip path:
        Identity, or 1x1 projection if spatial size / channel count changes

    Output:
        ReLU(main + skip)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = _make_norm(norm, out_channels, gn_groups=gn_groups)
        self.act1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = _make_norm(norm, out_channels, gn_groups=gn_groups)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _make_norm(norm, out_channels, gn_groups=gn_groups),
            )
        else:
            self.skip = nn.Identity()

        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act_out(out)
        return out


class ResPatchCenterCNN(nn.Module):
    """
    Patch-based residual CNN for center-cell multi-trait regression.

    Intended use:
        - Input:  raster patch of shape (B, C, H, W)
        - Output: trait vector for the center cell, shape (B, n_traits)

    Notes:
        - Input patch size must be odd (e.g. 15x15 or 25x25)
        - Output feature map size must also remain odd so the center location
          is well-defined
        - For 15x15 patches, a safe default is stride_blocks=(1, 1, 1, 1)
        - For 25x25 patches, stride_blocks=(1, 2, 1, 2) is also valid
    """

    def __init__(
        self,
        in_channels: int,
        n_traits: int = 31,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Four-stage residual encoder
        self.block1 = ResidualBlock(
            in_channels=in_channels,
            out_channels=c1,
            stride=stride_blocks[0],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )
        self.block2 = ResidualBlock(
            in_channels=c1,
            out_channels=c1,
            stride=stride_blocks[1],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )
        self.block3 = ResidualBlock(
            in_channels=c1,
            out_channels=c2,
            stride=stride_blocks[2],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )
        self.block4 = ResidualBlock(
            in_channels=c2,
            out_channels=c3,
            stride=stride_blocks[3],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )

        # Per-pixel trait prediction map; center cell is extracted in forward()
        self.head = nn.Conv2d(c3, n_traits, kernel_size=1, bias=True)

        self.in_channels = in_channels
        self.n_traits = n_traits
        self.base_channels = base_channels
        self.norm = norm
        self.gn_groups = gn_groups
        self.stride_blocks = stride_blocks
        self.dropout_p = dropout_p

    @staticmethod
    def _center_index(height: int, width: int) -> tuple[int, int]:
        """Return the unique center index of an odd-sized spatial map."""
        if height % 2 == 0 or width % 2 == 0:
            raise ValueError(
                "Output feature map must have odd spatial dimensions for true "
                f"center-cell prediction, got H={height}, W={width}."
            )
        return height // 2, width // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W), with odd H and W

        Returns:
            Tensor of shape (B, n_traits)
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected input of shape (B, C, H, W), got {tuple(x.shape)}"
            )

        _, _, height, width = x.shape
        if height % 2 == 0 or width % 2 == 0:
            raise ValueError(
                f"Input patch size must be odd, got H={height}, W={width}."
            )

        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)

        y_map = self.head(z)  # (B, n_traits, H_out, W_out)

        _, _, out_h, out_w = y_map.shape
        ci, cj = self._center_index(out_h, out_w)

        # Extract trait vector for the center cell only
        y_center = y_map[:, :, ci, cj]  # (B, n_traits)
        return y_center


if __name__ == "__main__":
    torch.manual_seed(0)

    model = ResPatchCenterCNN(
        in_channels=54,
        n_traits=31,
        base_channels=32,
        norm="gn",
        gn_groups=8,
        stride_blocks=(1, 1, 1, 1),  # safe for 15x15 patches
        # For 25x25 patches, can also use stride_blocks=(1, 2, 1, 2) for more downsampling
        dropout_p=0.1,
    )

    x = torch.randn(8, 54, 15, 15)
    y = model(x)

    print(model)
    print("Output shape:", y.shape)
