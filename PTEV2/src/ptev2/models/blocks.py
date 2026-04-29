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
        self.act1 = nn.PReLU(num_parameters=out_channels)
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

        self.act_out = nn.PReLU(num_parameters=out_channels)

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


class ResPatchCNN(nn.Module):
    """
    Patch-based residual CNN for multi-output regression with full spatial output.

    Intended use:
        - Input:  raster patch of shape (B, C, H, W)
        - Output: prediction map of shape (B, out_channels, H_out, W_out)

    Notes:
        - Output spatial size depends on stride_blocks configuration
        - For stride_blocks=(1, 1, 1, 1), output size ≈ input size
        - For stride_blocks with values > 1, output will be downsampled
        - Each output pixel predicts traits for the corresponding input region
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 31,
        n_traits: int | None = None,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
        head_depth: int = 2,
    ) -> None:
        super().__init__()
        # Backward compatibility: old configs/scripts may still pass n_traits.
        if n_traits is not None:
            out_channels = int(n_traits)

        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}.")

        if len(stride_blocks) != 4:
            raise ValueError(
                f"stride_blocks must have 4 entries, got {len(stride_blocks)}."
            )
        if any(s < 1 for s in stride_blocks):
            raise ValueError(f"All stride values must be >= 1, got {stride_blocks}.")
        stride_blocks = tuple(int(s) for s in stride_blocks)

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

        # Per-pixel prediction head (deeper head for better per-trait learning)
        head_layers = []
        in_features = c3
        for _ in range(max(1, head_depth - 1)):
            head_layers.append(
                nn.Conv2d(in_features, c3, kernel_size=3, padding=1, bias=False)
            )
            head_layers.append(_make_norm(norm, c3, gn_groups=gn_groups))
            head_layers.append(nn.PReLU(num_parameters=c3))
            head_layers.append(nn.Dropout2d(dropout_p))
            in_features = c3

        head_layers.append(
            nn.Conv2d(in_features, out_channels, kernel_size=1, bias=True)
        )
        self.head = nn.Sequential(*head_layers)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_traits = out_channels
        self.base_channels = base_channels
        self.norm = norm
        self.gn_groups = gn_groups
        self.stride_blocks = stride_blocks
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, out_channels, H_out, W_out)
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected input of shape (B, C, H, W), got {tuple(x.shape)}"
            )

        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)

        y_map = self.head(z)  # (B, out_channels, H_out, W_out)
        return y_map


if __name__ == "__main__":
    torch.manual_seed(0)

    model = ResPatchCNN(
        in_channels=54,
        out_channels=31,
        base_channels=32,
        norm="gn",
        gn_groups=8,
        stride_blocks=(1, 1, 1, 1),  # no downsampling
        dropout_p=0.1,
    )

    x = torch.randn(8, 54, 15, 15)
    y = model(x)

    print(model)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
