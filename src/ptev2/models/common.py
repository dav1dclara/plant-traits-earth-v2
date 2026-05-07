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


def _make_activation(
    activation: Literal["prelu", "silu", "relu", "gelu"],
    num_channels: int,
) -> nn.Module:
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


class SharedEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        activation: Literal["prelu", "silu", "gelu"] = "prelu",
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
            stride_blocks[0],
            norm,
            gn_groups,
            dropout_p,
            activation=activation,
        )
        self.block2 = ResidualBlock(
            c1,
            c1,
            stride_blocks[1],
            norm,
            gn_groups,
            dropout_p,
            activation=activation,
        )
        self.block3 = ResidualBlock(
            c1,
            c2,
            stride_blocks[2],
            norm,
            gn_groups,
            dropout_p,
            activation=activation,
        )
        self.block4 = ResidualBlock(
            c2,
            c3,
            stride_blocks[3],
            norm,
            gn_groups,
            dropout_p,
            activation=activation,
        )
        self.out_channels = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        return z


class TaskHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DeeperTaskHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        norm: Literal["bn", "gn", "none"] = "gn",
        activation: Literal["prelu", "silu", "gelu"] = "prelu",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, in_channels, gn_groups),
            _make_activation(activation, in_channels),
            nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def forward_with_intermediate(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inter = self.head[:-1](x)
        return self.head[-1](inter), inter


class DenseMMoELayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_experts: int = 4,
        expert_hidden: int = 64,
        n_tasks: int = 31,
        norm: Literal["bn", "gn", "none"] = "gn",
        activation: Literal["prelu", "silu", "gelu"] = "prelu",
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
                    activation=activation,
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


def load_balancing_loss(gate_weights: torch.Tensor, n_experts: int) -> torch.Tensor:
    f = gate_weights.mean(dim=(0, 1))
    return n_experts * (f * f).sum()
