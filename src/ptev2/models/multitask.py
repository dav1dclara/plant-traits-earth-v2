from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptev2.models.traitPatchCNN import ResidualBlock, _make_norm


class SharedEncoder(nn.Module):
    """Shared residual encoder used by the MTL and MMoE models."""

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

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

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
        z = self.block4(z)
        return z


class TaskHead(nn.Module):
    """Simple 1x1 prediction head."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class STLModel(nn.Module):
    """Single-task style dense model retained for compatibility."""

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
    ) -> None:
        super().__init__()
        if n_traits is not None:
            out_channels = int(n_traits)
        self.encoder = SharedEncoder(
            in_channels, base_channels, norm, gn_groups, stride_blocks, dropout_p
        )
        self.head = TaskHead(self.encoder.out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class MTLModel(nn.Module):
    """Shared encoder with one prediction head per output channel."""

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
    ) -> None:
        super().__init__()
        if n_traits is not None:
            out_channels = int(n_traits)
        self.encoder = SharedEncoder(
            in_channels, base_channels, norm, gn_groups, stride_blocks, dropout_p
        )
        self.heads = nn.ModuleList(
            [TaskHead(self.encoder.out_channels, 1) for _ in range(out_channels)]
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        outputs = [head(z) for head in self.heads]
        return torch.cat(outputs, dim=1)


class DenseMMoELayer(nn.Module):
    """Mixture-of-experts routing over dense spatial feature maps."""

    def __init__(
        self,
        in_features: int,
        n_experts: int = 4,
        expert_hidden: int = 64,
        n_tasks: int = 31,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
        gate_temperature: float = 1.0,
        gate_top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.gate_temperature = gate_temperature
        self.gate_top_k = gate_top_k

        self.experts = nn.ModuleList(
            [
                ResidualBlock(
                    in_features,
                    expert_hidden,
                    stride=1,
                    norm=norm,
                    gn_groups=gn_groups,
                    dropout_p=dropout_p,
                )
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

    def _apply_gate_constraints(self, gate_logits: torch.Tensor) -> torch.Tensor:
        scaled_logits = gate_logits / max(self.gate_temperature, 1e-6)

        if self.gate_top_k is None or self.gate_top_k >= self.n_experts:
            return F.softmax(scaled_logits, dim=-1)

        top_k = max(1, self.gate_top_k)
        top_values, top_indices = torch.topk(scaled_logits, k=top_k, dim=-1)
        sparse_logits = torch.full_like(scaled_logits, float("-inf"))
        sparse_logits.scatter_(-1, top_indices, top_values)
        return F.softmax(sparse_logits, dim=-1)

    def forward(
        self,
        spatial_features: torch.Tensor,
        route_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = [expert(spatial_features) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)

        gate_logits = torch.stack([gate(route_features) for gate in self.gates], dim=1)
        gate_weights = self._apply_gate_constraints(gate_logits)
        task_maps = torch.einsum("bte,bechw->btchw", gate_weights, expert_stack)
        return task_maps, gate_weights


class MMoEModel(nn.Module):
    """Routed dense MMoE model with shared projection and gated expert mixing."""

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
        n_experts: int = 4,
        expert_hidden: int = 64,
        gate_temperature: float = 1.0,
        gate_top_k: int | None = None,
    ) -> None:
        super().__init__()
        if n_traits is not None:
            out_channels = int(n_traits)

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

        self.shared_projection = nn.Sequential(
            nn.Conv2d(c3, expert_hidden, kernel_size=1, bias=False),
            _make_norm(norm, expert_hidden, gn_groups),
            nn.PReLU(num_parameters=expert_hidden),
        )

        self.heads = nn.ModuleList(
            [TaskHead(expert_hidden, 1) for _ in range(out_channels)]
        )
        self.out_channels = out_channels
        self.last_gate_weights: torch.Tensor | None = None
        self.gate_temperature = gate_temperature
        self.gate_top_k = gate_top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        route_features = self.route_pool(z).flatten(1)
        task_maps, gate_weights = self.mmoe(z, route_features)
        shared_map = self.shared_projection(z)
        self.last_gate_weights = gate_weights.detach()

        outputs = []
        for idx, head in enumerate(self.heads):
            task_features = task_maps[:, idx] + shared_map
            outputs.append(head(task_features))
        return torch.cat(outputs, dim=1)


class GatedMMoEModel(MMoEModel):
    """Alias for explicit config naming of the routed MMoE architecture."""
