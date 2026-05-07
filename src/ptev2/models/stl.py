from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ptev2.models.common import SharedEncoder, TaskHead


class STLModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 31,
        n_traits: int | None = None,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        activation: Literal["prelu", "silu", "gelu"] = "prelu",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if n_traits is not None:
            out_channels = int(n_traits)
        self.encoder = SharedEncoder(
            in_channels,
            base_channels,
            norm,
            activation,
            gn_groups,
            stride_blocks,
            dropout_p,
        )
        self.head = TaskHead(self.encoder.out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
