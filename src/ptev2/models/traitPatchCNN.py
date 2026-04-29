from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _make_activation(
    activation: Literal["prelu", "silu", "relu", "gelu"],
    num_channels: int,
) -> nn.Module:
    """Create activation layer."""
    if activation == "prelu":
        return nn.PReLU(num_parameters=num_channels)
    if activation == "silu":
        return nn.SiLU()
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation='{activation}'")


class ResidualBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions.

    Main path:
        Conv3x3 -> Norm -> Act -> Dropout -> Conv3x3 -> Norm

    Skip path:
        Identity, or 1x1 projection if spatial size / channel count changes

    Output:
        Act(main + skip)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
        dilation: int = 1,
        activation: Literal["prelu", "silu", "relu", "gelu"] = "prelu",
    ) -> None:
        super().__init__()
        padding = dilation

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = _make_norm(norm, out_channels, gn_groups=gn_groups)
        self.act1 = _make_activation(activation, out_channels)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation,
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

        self.act_out = _make_activation(activation, out_channels)

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
        - Output spatial size can be controlled via output_mode.
        - For small chips (e.g. 7x7), keep strides conservative and use dilation for context.
        - For larger chips, encoder downsampling is supported while preserving output grid.
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
        encoder_dilations: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
        head_depth: int = 2,
        activation: Literal["prelu", "silu", "relu", "gelu"] = "prelu",
        use_stem: bool = True,
        stem_kernel_size: int = 3,
        output_mode: Literal["native", "upsample_to_input"] = "upsample_to_input",
        use_projection: bool = True,
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
        if len(encoder_dilations) != 4:
            raise ValueError(
                f"encoder_dilations must have 4 entries, got {len(encoder_dilations)}."
            )
        if any(d < 1 for d in encoder_dilations):
            raise ValueError(
                f"All dilation values must be >= 1, got {encoder_dilations}."
            )
        encoder_dilations = tuple(int(d) for d in encoder_dilations)
        if stem_kernel_size < 1 or stem_kernel_size % 2 == 0:
            raise ValueError(
                f"stem_kernel_size must be odd and >=1, got {stem_kernel_size}."
            )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.use_stem = bool(use_stem)
        if self.use_stem:
            stem_padding = stem_kernel_size // 2
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    c1,
                    kernel_size=stem_kernel_size,
                    padding=stem_padding,
                    bias=False,
                ),
                _make_norm(norm, c1, gn_groups=gn_groups),
                _make_activation(activation, c1),
            )
            block1_in_channels = c1
        else:
            self.stem = nn.Identity()
            block1_in_channels = in_channels

        # Four-stage residual encoder
        self.block1 = ResidualBlock(
            in_channels=block1_in_channels,
            out_channels=c1,
            stride=stride_blocks[0],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
            dilation=encoder_dilations[0],
            activation=activation,
        )
        self.block2 = ResidualBlock(
            in_channels=c1,
            out_channels=c1,
            stride=stride_blocks[1],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
            dilation=encoder_dilations[1],
            activation=activation,
        )
        self.block3 = ResidualBlock(
            in_channels=c1,
            out_channels=c2,
            stride=stride_blocks[2],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
            dilation=encoder_dilations[2],
            activation=activation,
        )
        self.block4 = ResidualBlock(
            in_channels=c2,
            out_channels=c3,
            stride=stride_blocks[3],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
            dilation=encoder_dilations[3],
            activation=activation,
        )

        if use_projection:
            self.projection = nn.Sequential(
                nn.Conv2d(c3, c3, kernel_size=1, bias=False),
                _make_norm(norm, c3, gn_groups=gn_groups),
                _make_activation(activation, c3),
            )
        else:
            self.projection = nn.Identity()

        # Per-pixel prediction head (deeper head for better per-trait learning)
        head_layers = []
        in_features = c3
        for _ in range(max(1, head_depth - 1)):
            head_layers.append(
                nn.Conv2d(in_features, c3, kernel_size=3, padding=1, bias=False)
            )
            head_layers.append(_make_norm(norm, c3, gn_groups=gn_groups))
            head_layers.append(_make_activation(activation, c3))
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
        self.encoder_dilations = encoder_dilations
        self.dropout_p = dropout_p
        self.activation = activation
        self.output_mode = output_mode
        self.use_projection = bool(use_projection)

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

        input_hw = x.shape[-2:]
        z = self.stem(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        z = self.projection(z)

        y_map = self.head(z)  # (B, out_channels, H_out, W_out)
        if self.output_mode == "upsample_to_input" and y_map.shape[-2:] != input_hw:
            y_map = F.interpolate(
                y_map,
                size=input_hw,
                mode="bilinear",
                align_corners=False,
            )
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
        encoder_dilations=(1, 1, 2, 2),
        activation="silu",
        dropout_p=0.1,
    )

    x = torch.randn(8, 54, 15, 15)
    y = model(x)

    print(model)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
