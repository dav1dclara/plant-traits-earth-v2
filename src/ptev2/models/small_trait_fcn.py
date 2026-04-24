from __future__ import annotations

import torch
import torch.nn as nn


def _group_norm(num_channels: int, num_groups: int) -> nn.GroupNorm:
    groups = min(num_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_groups: int = 8,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = _group_norm(channels, num_groups)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = _group_norm(channels, num_groups)
        self.act_out = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act_out(out + x)


class _TraitHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_groups: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            _group_norm(hidden_channels, num_groups),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallTraitFCN(nn.Module):
    """Small fully convolutional residual model for center-supervised multi-trait regression."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        num_groups: int = 8,
        head_mode: str = "joint",
        trait_head_hidden_channels: int = 16,
        block3_dilation: int = 1,
    ) -> None:
        super().__init__()
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}.")
        head_mode = str(head_mode).lower()
        if head_mode not in {"joint", "trait_heads"}:
            raise ValueError(
                f"Unsupported head_mode '{head_mode}'. Expected one of ['joint', 'trait_heads']."
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(base_channels, num_groups),
            nn.SiLU(),
        )

        self.block1 = _ResidualBlock(base_channels, num_groups=num_groups, dilation=1)
        self.block2 = _ResidualBlock(base_channels, num_groups=num_groups, dilation=1)
        self.block3 = _ResidualBlock(
            base_channels,
            num_groups=num_groups,
            dilation=int(block3_dilation),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=1, bias=False),
            _group_norm(base_channels, num_groups),
            nn.SiLU(),
        )

        self.head_mode = head_mode
        self.out_channels = int(out_channels)
        if self.head_mode == "joint":
            self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.heads = nn.ModuleList(
                [
                    _TraitHead(
                        in_channels=base_channels,
                        hidden_channels=int(trait_head_hidden_channels),
                        num_groups=num_groups,
                    )
                    for _ in range(out_channels)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor with shape (B, C, H, W), got {tuple(x.shape)}"
            )
        z = self.stem(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        z = self.projection(z)

        if self.head_mode == "joint":
            return self.head(z)
        return torch.cat([head(z) for head in self.heads], dim=1)
