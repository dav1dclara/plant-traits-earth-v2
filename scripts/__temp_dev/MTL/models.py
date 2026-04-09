from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(
    norm: Literal["bn", "gn", "none"], num_channels: int, gn_groups: int = 8
) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        groups = min(gn_groups, num_channels)
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    return nn.Identity()


class ResidualBlock(nn.Module):
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
        self.norm1 = _make_norm(norm, out_channels, gn_groups)
        self.act1 = nn.PReLU(num_parameters=out_channels)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = _make_norm(norm, out_channels, gn_groups)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
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
        out = out + identity
        return self.act_out(out)


class ResPatchCNN(nn.Module):
    """Backbone used by STL and MTL models."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 31,
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
            in_channels,
            c1,
            stride=stride_blocks[0],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )
        self.block2 = ResidualBlock(
            c1,
            c1,
            stride=stride_blocks[1],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )
        self.block3 = ResidualBlock(
            c1,
            c2,
            stride=stride_blocks[2],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )
        self.block4 = ResidualBlock(
            c2,
            c3,
            stride=stride_blocks[3],
            norm=norm,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
        )

        self.head = nn.Conv2d(c3, out_channels, kernel_size=1, bias=True)

        self.out_channels = out_channels
        self.base_channels = base_channels
        self.norm = norm
        self.gn_groups = gn_groups
        self.stride_blocks = stride_blocks
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        return self.head(z)


class MTLModel(nn.Module):
    """Backbone + one small head per trait."""

    def __init__(
        self,
        in_channels: int,
        n_traits: int,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = ResPatchCNN(
            in_channels=in_channels,
            out_channels=base_channels * 4,
            base_channels=base_channels,
            norm=norm,
            gn_groups=gn_groups,
            stride_blocks=stride_blocks,
            dropout_p=dropout_p,
        )
        head_channels = base_channels * 4
        self.heads = nn.ModuleList(
            [nn.Conv2d(head_channels, 1, kernel_size=1) for _ in range(n_traits)]
        )
        self.n_traits = n_traits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.block1(x)
        features = self.backbone.block2(features)
        features = self.backbone.block3(features)
        features = self.backbone.block4(features)

        outputs = [head(features) for head in self.heads]
        return torch.cat(outputs, dim=1)


class MMoELayer(nn.Module):
    """Simple spatial MMoE layer that operates per pixel."""

    def __init__(
        self,
        in_dim: int,
        n_experts: int,
        expert_dim: int,
        n_tasks: int,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(in_dim, expert_dim) for _ in range(n_experts)]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(in_dim, n_experts) for _ in range(n_tasks)]
        )
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.expert_dim = expert_dim

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: (batch_pixels, in_dim)
        expert_outputs = torch.stack(
            [F.relu(expert(x)) for expert in self.experts], dim=1
        )
        task_outputs: list[torch.Tensor] = []
        for gate in self.gates:
            weights = F.softmax(gate(x), dim=-1)
            gated = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
            task_outputs.append(gated)
        return task_outputs


class MMoEModel(nn.Module):
    """Backbone + MMoE + one output head per trait."""

    def __init__(
        self,
        in_channels: int,
        n_traits: int,
        base_channels: int = 32,
        n_experts: int = 4,
        expert_dim: int = 64,
        tower_hidden: int = 64,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = ResPatchCNN(
            in_channels=in_channels,
            out_channels=base_channels * 4,
            base_channels=base_channels,
            norm=norm,
            gn_groups=gn_groups,
            stride_blocks=stride_blocks,
            dropout_p=dropout_p,
        )
        self.n_traits = n_traits
        self.n_experts = n_experts
        self.expert_dim = expert_dim
        self.gate_dim = base_channels * 4

        self.mmoe = MMoELayer(
            in_dim=self.gate_dim,
            n_experts=n_experts,
            expert_dim=expert_dim,
            n_tasks=n_traits,
        )
        self.tower_heads = nn.ModuleList(
            [nn.Linear(expert_dim, 1) for _ in range(n_traits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.block1(x)
        features = self.backbone.block2(features)
        features = self.backbone.block3(features)
        features = self.backbone.block4(features)

        B, C, H, W = features.shape
        pixels = features.permute(0, 2, 3, 1).reshape(-1, C)
        mmoe_outputs = self.mmoe(pixels)

        results = [
            head(task_feat).view(B, 1, H, W)
            for task_feat, head in zip(mmoe_outputs, self.tower_heads)
        ]
        return torch.cat(results, dim=1)


STLModel = ResPatchCNN


def build_model(model_type: str, in_channels: int, n_tasks: int, **kwargs) -> nn.Module:
    if model_type == "stl":
        return STLModel(in_channels=in_channels, out_channels=n_tasks, **kwargs)
    if model_type == "mtl":
        return MTLModel(in_channels=in_channels, n_traits=n_tasks, **kwargs)
    if model_type == "mmoe":
        return MMoEModel(in_channels=in_channels, n_traits=n_tasks, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")
