from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ptev2.models.traitPatchCNN import ResidualBlock


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
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks

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
                    nn.Softmax(dim=-1),
                )
                for _ in range(n_tasks)
            ]
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        route_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = [expert(spatial_features) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)
        gate_weights = torch.stack([gate(route_features) for gate in self.gates], dim=1)
        task_maps = torch.einsum("bte,bechw->btchw", gate_weights, expert_stack)
        return task_maps, gate_weights


class MMoEModel(nn.Module):
    """Dense MMoE model with shared encoder, experts, and task-specific heads."""

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
        )
        self.heads = nn.ModuleList(
            [TaskHead(expert_hidden, 1) for _ in range(out_channels)]
        )
        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        route_features = self.route_pool(z).flatten(1)
        task_maps, gate_weights = self.mmoe(z, route_features)
        self.last_gate_weights = gate_weights.detach()
        outputs = [head(task_maps[:, idx]) for idx, head in enumerate(self.heads)]
        return torch.cat(outputs, dim=1)
