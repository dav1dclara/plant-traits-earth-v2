from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ptev2.models.common import (
    DeeperTaskHead,
    SharedEncoder,
    TaskHead,
    _make_activation,
    _make_norm,
)


class MTLModel(nn.Module):
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
        per_trait_conv: bool = False,
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
        c3 = self.encoder.out_channels
        self.per_trait_conv = per_trait_conv

        if per_trait_conv:
            self.shared_head = None
            self.heads = nn.ModuleList(
                [
                    DeeperTaskHead(c3, 1, norm, activation, gn_groups, dropout_p)
                    for _ in range(out_channels)
                ]
            )
        else:
            self.shared_head = nn.Sequential(
                nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
                _make_norm(norm, c3, gn_groups),
                _make_activation(activation, c3),
                nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity(),
            )
            self.heads = nn.ModuleList([TaskHead(c3, 1) for _ in range(out_channels)])
        self.out_channels = out_channels
        self._last_intermediates: list[torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.shared_head is not None:
            z = self.shared_head(z)
        if self.per_trait_conv and self.training:
            outputs: list[torch.Tensor] = []
            intermediates: list[torch.Tensor] = []
            for head in self.heads:
                out, inter = head.forward_with_intermediate(z)
                outputs.append(out)
                intermediates.append(inter)
            self._last_intermediates = intermediates
        else:
            self._last_intermediates = None
            outputs = [head(z) for head in self.heads]
        return torch.cat(outputs, dim=1)

    def get_group_consistency_loss(
        self, trait_group_indices: list[list[int]]
    ) -> torch.Tensor:
        if self._last_intermediates is None or not trait_group_indices:
            return torch.tensor(0.0)
        total = self._last_intermediates[0].new_zeros(())
        n = 0
        for idx_list in trait_group_indices:
            if len(idx_list) < 2:
                continue
            g = torch.stack([self._last_intermediates[i] for i in idx_list], dim=1)
            mean_g = g.mean(dim=1, keepdim=True)
            var = ((g - mean_g) ** 2).mean()
            total = total + var
            n += 1
        return total / max(n, 1)
