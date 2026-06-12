"""Trait-grouped Gated MMoE model for plant-trait prediction.

This module contains GatedMMoEModelV3, the architecture motivated by the
Ward-cluster analysis in trait_correlation_analysis.ipynb.

Design rationale
----------------
All 37 traits are routed through the shared 8-expert MMoE. Each trait has
its own gating network that learns which combination of experts to weight,
allowing coherent trait groups to converge on shared experts while
structurally unrelated traits route independently.

The 8 experts correspond to the 8 Ward clusters (d = 1 − |r|):
  C0  Leaf water & structural density : X3120, X4, X47, X6
  C1  Plant size & life history       : X1080, X21, X237, X26, X27, X3106, X3107, X614, X95
  C2  Leaf size & mass  (TIGHT)       : X144, X145, X163, X3112, X3113, X3114, X55   |r|=0.793
  C3  Seed number & vascular anatomy  : X138, X169, X224, X281
  C4  Leaf economics spectrum (TIGHT) : X3117, X46, X50                               |r|=0.749
  C5  Leaf nutrient stoichiometry     : X13, X14, X146, X15, X78
  C6  Wood anatomy lengths            : X282, X289
  C7  Structurally unrelated residuals: X223, X297, X351

Architecture
------------
  SharedEncoder                     shared 4-block residual backbone
  └── DenseMMoELayer (8 experts)    routes all 37 traits
        └── DeeperTaskHead × 37    per-trait prediction head
"""

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
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        g = min(gn_groups, num_channels)
        while g > 1 and num_channels % g != 0:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm='{norm}'")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = _make_norm(norm, out_channels, gn_groups)
        self.act1 = nn.PReLU(num_parameters=out_channels)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = _make_norm(norm, out_channels, gn_groups)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _make_norm(norm, out_channels, gn_groups),
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
        return self.act_out(out + identity)


class SharedEncoder(nn.Module):
    """Four-block residual encoder shared across all traits."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.block1 = ResidualBlock(
            in_channels, c1, stride_blocks[0], norm, gn_groups, dropout_p
        )
        self.block2 = ResidualBlock(
            c1, c1, stride_blocks[1], norm, gn_groups, dropout_p
        )
        self.block3 = ResidualBlock(
            c1, c2, stride_blocks[2], norm, gn_groups, dropout_p
        )
        self.block4 = ResidualBlock(
            c2, c3, stride_blocks[3], norm, gn_groups, dropout_p
        )
        self.out_channels = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        return self.block4(z)


class DenseMMoELayer(nn.Module):
    """Dense MMoE with temperature-scaled, optionally sparse gates."""

    def __init__(
        self,
        in_features: int,
        n_experts: int = 8,
        expert_hidden: int = 96,
        n_tasks: int = 37,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
        gate_temperature: float = 0.8,
        gate_top_k: int | None = 2,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.gate_temperature = float(gate_temperature)
        self.gate_top_k = gate_top_k
        self.experts = nn.ModuleList(
            [
                ResidualBlock(in_features, expert_hidden, 1, norm, gn_groups, dropout_p)
                for _ in range(n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, in_features),
                    nn.ReLU(),
                    nn.Linear(in_features, n_experts),
                )
                for _ in range(n_tasks)
            ]
        )

    def _gate_weights(self, logits: torch.Tensor) -> torch.Tensor:
        scaled = logits / max(self.gate_temperature, 1e-6)
        if self.gate_top_k is None or self.gate_top_k >= self.n_experts:
            return F.softmax(scaled, dim=-1)
        k = max(1, self.gate_top_k)
        top_vals, top_idx = torch.topk(scaled, k, dim=-1)
        sparse = torch.full_like(scaled, float("-inf"))
        sparse.scatter_(-1, top_idx, top_vals)
        return F.softmax(sparse, dim=-1)

    def forward(
        self,
        spatial: torch.Tensor,
        route: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expert_stack = torch.stack([e(spatial) for e in self.experts], dim=1)
        gate_logits = torch.stack([g(route) for g in self.gates], dim=1)
        gate_w = self._gate_weights(gate_logits)
        task_maps = torch.einsum("bte,bechw->btchw", gate_w, expert_stack)
        return task_maps, gate_w


class DeeperTaskHead(nn.Module):
    """3x3 conv + 1x1 conv head."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            _make_norm(norm, in_channels, gn_groups),
            nn.PReLU(num_parameters=in_channels),
            nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity(),
        )
        self.out = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


class GatedMMoEModelV3(nn.Module):
    """Gated MMoE with all 37 traits routed through shared experts.

    All traits pass through the same 8-expert MMoE layer. Each trait has
    its own gating network that learns which combination of experts to use.
    Traits within coherent Ward clusters are expected to converge on similar
    expert combinations; structurally unrelated traits (G7) route differently
    without any explicit constraint.

    Args:
        in_channels: total predictor channels (e.g. 150).
        out_channels: total traits to predict (default 37).
        base_channels: encoder width; doubles each stage (32 -> 64 -> 128).
        norm: normalisation type ('gn', 'bn', or 'none').
        gn_groups: groups for GroupNorm.
        stride_blocks: spatial stride per encoder block.
        dropout_p: dropout probability throughout.
        n_experts: number of MMoE experts (default 8, one per Ward cluster).
        expert_hidden: channel width for experts and task heads.
        gate_temperature: softmax temperature (<1 sharpens, >1 flattens).
        gate_top_k: active experts per trait per forward pass (None = all).
        memory_efficient: if True, runs experts one at a time to save memory.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 37,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.1,
        n_experts: int = 8,
        expert_hidden: int = 64,
        gate_temperature: float = 0.8,
        gate_top_k: int | None = 2,
        memory_efficient: bool = True,
    ) -> None:
        super().__init__()
        self._out_channels = out_channels
        self.memory_efficient = bool(memory_efficient)

        self.encoder = SharedEncoder(
            in_channels, base_channels, norm, gn_groups, stride_blocks, dropout_p
        )
        c3 = self.encoder.out_channels

        self.route_pool = nn.AdaptiveAvgPool2d(1)
        self.mmoe = DenseMMoELayer(
            c3,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            n_tasks=out_channels,
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
            gate_temperature=gate_temperature,
            gate_top_k=gate_top_k,
        )

        self.heads = nn.ModuleList(
            [
                DeeperTaskHead(expert_hidden, 1, norm, gn_groups, dropout_p)
                for _ in range(out_channels)
            ]
        )

        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        route = self.route_pool(z).flatten(1)

        if self.memory_efficient:
            gate_logits = torch.stack([g(route) for g in self.mmoe.gates], dim=1)
            gate_w = self.mmoe._gate_weights(gate_logits)
            self.last_gate_weights = gate_w.detach()

            outputs = []
            for i, head in enumerate(self.heads):
                feat = None
                for expert_idx, expert in enumerate(self.mmoe.experts):
                    expert_feat = expert(z)
                    weighted = gate_w[:, i, expert_idx].view(-1, 1, 1, 1) * expert_feat
                    feat = weighted if feat is None else feat + weighted
                outputs.append(head(feat))
        else:
            task_maps, gate_w = self.mmoe(z, route)
            self.last_gate_weights = gate_w.detach()
            outputs = [self.heads[i](task_maps[:, i]) for i in range(self._out_channels)]

        return torch.cat(outputs, dim=1)  # (B, T, H, W)
