from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ptev2.models.common import _make_activation, _make_norm


class PixelMLP(nn.Module):
    """Tiny per-pixel MLP implemented with 1x1 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 31,
        n_traits: int | None = None,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout_p: float = 0.05,
        norm: Literal["bn", "gn", "none"] = "gn",
        activation: Literal["prelu", "silu", "gelu"] = "prelu",
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        if n_traits is not None:
            out_channels = int(n_traits)

        if int(depth) <= 0:
            self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            layers: list[nn.Module] = []
            prev = in_channels
            for _ in range(int(depth)):
                layers.append(nn.Conv2d(prev, hidden_dim, kernel_size=1, bias=False))
                layers.append(_make_norm(norm, hidden_dim, gn_groups))
                layers.append(_make_activation(activation, hidden_dim))
                if float(dropout_p) > 0.0:
                    layers.append(nn.Dropout2d(float(dropout_p)))
                prev = hidden_dim
            layers.append(nn.Conv2d(prev, out_channels, kernel_size=1, bias=True))
            self.net = nn.Sequential(*layers)

        if int(in_channels) == 150 and int(out_channels) == 31:
            param_count = sum(p.numel() for p in self.parameters())
            print(f"PixelMLP param count: {param_count:,} (expected ~16k for 150→31)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
