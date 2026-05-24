"""Trait-grouped Gated MMoE model for plant-trait prediction.

This module contains GatedMMoEModelV3, the architecture motivated by the
Ward-cluster analysis in trait_correlation_analysis.ipynb.

Design rationale
----------------
Two findings from the analysis drive the architecture:

1. Hard traits (X223, X282, X289, X297) have near-zero inter-trait correlation
   AND near-zero EO predictability (test r ≈ 0). Routing them through the shared
   MMoE pollutes the experts with unlearnable signal. They get a private bypass
   path (ResidualBlock → head) with no cross-task routing.

2. The remaining 33 traits are grouped into 8 Ward clusters (d = 1 − |r|,
   cut-off 0.45). Two clusters are especially tight and benefit most from
   the shared experts:
     C2 Leaf size / mass  (X144, X145, X163, X3112, X3113, X3114, X55)  |r|=0.793
     C4 Leaf structure    (X3117, X46, X50)                              |r|=0.749
   One expert per Ward cluster (n_experts=8) gives the gating sufficient
   capacity to specialise per functional group.

Architecture
------------
  SharedEncoder                        shared 4-block residual backbone
  ├── DenseMMoELayer (8 experts)        routes the 33 soft traits
  │     └── DeeperTaskHead × 33        per-trait prediction head
  └── BypassBlock × 4 + Head × 4       private path for the 4 hard traits

Ward clusters (trait_correlation_analysis.ipynb, 8-cluster Ward linkage)
-------------------------------------------------------------------------
C0  Water / root mix          : X3120, X4, X47, X6
C1  Size / propagule mix      : X1080, X21, X237, X26, X27, X3106, X3107, X614, X95
C2  Leaf size / mass  (TIGHT) : X144, X145, X163, X3112, X3113, X3114, X55   |r|=0.793
C3  Seed / conduit mix        : X138, X169, X224, X281
C4  Leaf structure   (TIGHT)  : X3117, X46, X50                               |r|=0.749
C5  Leaf chemistry            : X13, X14, X146, X15, X78
C6  Wood-length pair          : X282, X289   (bypassed — hard EO targets)
C7  Independent / hard        : X223, X297, X351
     └─ X223, X282, X289, X297 → bypass   X351 → soft MMoE (lower priority)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hard trait bypass indices  (0-based, in the 37-trait TRAIT_NAMES order)
# X223=11, X282=17, X289=18, X297=19
# ---------------------------------------------------------------------------
_DEFAULT_HARD_INDICES: tuple[int, ...] = (11, 17, 18, 19)

# ---------------------------------------------------------------------------
# Low-level building blocks (self-contained, no external ptev2 dependencies)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Higher-level blocks
# ---------------------------------------------------------------------------


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
        n_experts: int = 4,
        expert_hidden: int = 96,
        n_tasks: int = 37,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
        gate_temperature: float = 0.4,
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
    """3×3 conv + 1×1 conv head."""

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


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class GatedMMoEModelV3(nn.Module):
    """Trait-grouped Gated MMoE with hard-trait bypass (analysis-motivated V3).

    33 soft traits are routed through the 8-expert MMoE (one expert per Ward
    cluster). 4 hard traits (X223, X282, X289, X297) bypass the MMoE and each
    receive a private ResidualBlock + head — no cross-task routing.

    Args:
        in_channels: total predictor channels (e.g. 150 for patch7_stride3).
        out_channels: total traits to predict.
        base_channels: encoder width; doubles each stage (48 → 96 → 192).
        norm: normalisation type ('gn', 'bn', or 'none').
        gn_groups: groups for GroupNorm.
        stride_blocks: spatial stride per encoder block.
        dropout_p: dropout probability throughout.
        n_experts: number of MMoE experts (default 8, one per Ward cluster).
        expert_hidden: channel width for experts and task heads.
        gate_temperature: softmax temperature (<1 sharpens, >1 flattens).
        gate_top_k: active experts per soft trait (None = all).
        residual_scale: if > 0, adds a scaled shared encoder projection to
            every soft task map before the head.
        hard_trait_indices: 0-based trait indices that bypass the MMoE.
            Default: (11, 17, 18, 19) = X223, X282, X289, X297.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 37,
        base_channels: int = 48,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.1,
        n_experts: int = 8,
        expert_hidden: int = 96,
        gate_temperature: float = 0.8,
        gate_top_k: int | None = 2,
        residual_scale: float = 0.0,
        hard_trait_indices: tuple[int, ...] = _DEFAULT_HARD_INDICES,
        memory_efficient: bool = True,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self._out_channels = out_channels
        self.memory_efficient = bool(memory_efficient)

        hard_set = set(hard_trait_indices)
        self.hard_indices: list[int] = sorted(hard_set)
        self.soft_indices: list[int] = [
            i for i in range(out_channels) if i not in hard_set
        ]
        n_soft = len(self.soft_indices)
        n_hard = len(self.hard_indices)

        self.encoder = SharedEncoder(
            in_channels, base_channels, norm, gn_groups, stride_blocks, dropout_p
        )
        c3 = self.encoder.out_channels

        self.route_pool = nn.AdaptiveAvgPool2d(1)
        self.mmoe = DenseMMoELayer(
            c3,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            n_tasks=n_soft,
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
            gate_temperature=gate_temperature,
            gate_top_k=gate_top_k,
        )

        if residual_scale > 0.0:
            self.shared_proj = nn.Sequential(
                nn.Conv2d(c3, expert_hidden, 1, bias=False),
                _make_norm(norm, expert_hidden, gn_groups),
                nn.PReLU(num_parameters=expert_hidden),
            )
        else:
            self.shared_proj = None

        self.soft_heads = nn.ModuleList(
            [
                DeeperTaskHead(expert_hidden, 1, norm, gn_groups, dropout_p)
                for _ in range(n_soft)
            ]
        )
        self.bypass_blocks = nn.ModuleList(
            [
                ResidualBlock(c3, expert_hidden, 1, norm, gn_groups, dropout_p)
                for _ in range(n_hard)
            ]
        )
        self.hard_heads = nn.ModuleList(
            [
                DeeperTaskHead(expert_hidden, 1, norm, gn_groups, dropout_p)
                for _ in range(n_hard)
            ]
        )

        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)

        route = self.route_pool(z).flatten(1)
        if self.memory_efficient:
            gate_logits = torch.stack([g(route) for g in self.mmoe.gates], dim=1)
            gate_w = self.mmoe._gate_weights(gate_logits)
            task_maps = None
        else:
            task_maps, gate_w = self.mmoe(z, route)
        self.last_gate_weights = gate_w.detach()

        shared = self.shared_proj(z) if self.shared_proj is not None else None

        soft_outs: list[torch.Tensor] = []
        for i, head in enumerate(self.soft_heads):
            if task_maps is None:
                feat = None
                for expert_idx, expert in enumerate(self.mmoe.experts):
                    expert_feat = expert(z)
                    weighted = (
                        gate_w[:, i, expert_idx].view(-1, 1, 1, 1) * expert_feat
                    )
                    feat = weighted if feat is None else feat + weighted
                if feat is None:
                    raise RuntimeError("MMoE has no experts.")
            else:
                feat = task_maps[:, i]
            if shared is not None and self.residual_scale > 0.0:
                feat = feat + self.residual_scale * shared
            soft_outs.append(head(feat))

        hard_outs: list[torch.Tensor] = [
            head(block(z)) for block, head in zip(self.bypass_blocks, self.hard_heads)
        ]

        out: list[torch.Tensor | None] = [None] * self._out_channels
        for i, idx in enumerate(self.soft_indices):
            out[idx] = soft_outs[i]
        for i, idx in enumerate(self.hard_indices):
            out[idx] = hard_outs[i]

        return torch.cat(out, dim=1)  # (B, T, H, W)
