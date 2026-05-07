from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ptev2.models.common import (
    DeeperTaskHead,
    DenseMMoELayer,
    SharedEncoder,
    TaskHead,
    _make_activation,
    _make_norm,
    load_balancing_loss,
)


class GatedMMoEModel(nn.Module):
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
        n_experts: int = 4,
        expert_hidden: int = 64,
        gate_temperature: float = 1.0,
        gate_top_k: int | None = None,
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

        self.route_pool = nn.AdaptiveAvgPool2d(1)
        self.mmoe = DenseMMoELayer(
            c3,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            n_tasks=out_channels,
            norm=norm,
            activation=activation,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
            gate_temperature=gate_temperature,
            gate_top_k=gate_top_k,
        )

        self.shared_projection = nn.Sequential(
            nn.Conv2d(c3, expert_hidden, kernel_size=1, bias=False),
            _make_norm(norm, expert_hidden, gn_groups),
            _make_activation(activation, expert_hidden),
        )

        self.heads = nn.ModuleList(
            [TaskHead(expert_hidden, 1) for _ in range(out_channels)]
        )
        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        route_features = self.route_pool(z).flatten(1)
        task_maps, gate_weights = self.mmoe(z, route_features)
        shared_map = self.shared_projection(z)
        self.last_gate_weights = gate_weights.detach()

        outputs = []
        for i, head in enumerate(self.heads):
            task_features = task_maps[:, i] + shared_map
            outputs.append(head(task_features))

        return torch.cat(outputs, dim=1)


class GatedMMoEModelV3(nn.Module):
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
        n_experts: int = 6,
        expert_hidden: int = 192,
        gate_temperature: float = 1.0,
        gate_top_k: int | None = 4,
        residual_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if n_traits is not None:
            out_channels = int(n_traits)

        self.residual_scale = residual_scale
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
        self.route_pool = nn.AdaptiveAvgPool2d(1)
        self.mmoe = DenseMMoELayer(
            c3,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            n_tasks=out_channels,
            norm=norm,
            activation=activation,
            gn_groups=gn_groups,
            dropout_p=dropout_p,
            gate_temperature=gate_temperature,
            gate_top_k=gate_top_k,
        )

        if residual_scale > 0:
            self.shared_projection = nn.Sequential(
                nn.Conv2d(c3, expert_hidden, kernel_size=1, bias=False),
                _make_norm(norm, expert_hidden, gn_groups),
                _make_activation(activation, expert_hidden),
            )
        else:
            self.shared_projection = None

        self.heads = nn.ModuleList(
            [
                DeeperTaskHead(
                    expert_hidden,
                    1,
                    norm,
                    activation,
                    gn_groups,
                    dropout_p,
                )
                for _ in range(out_channels)
            ]
        )

        self.last_gate_weights: torch.Tensor | None = None
        self._gate_weights_grad: torch.Tensor | None = None
        self._n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        route_features = self.route_pool(z).flatten(1)
        task_maps, gate_weights = self.mmoe(z, route_features)
        self.last_gate_weights = gate_weights.detach()
        self._gate_weights_grad = gate_weights

        if self.shared_projection is not None and self.residual_scale > 0:
            shared_map = self.shared_projection(z)
        else:
            shared_map = None

        outputs = []
        for i, head in enumerate(self.heads):
            task_features = task_maps[:, i]
            if shared_map is not None:
                task_features = task_features + self.residual_scale * shared_map
            outputs.append(head(task_features))

        return torch.cat(outputs, dim=1)

    def get_load_balancing_loss(self) -> torch.Tensor:
        gw = (
            self._gate_weights_grad
            if self._gate_weights_grad is not None
            else self.last_gate_weights
        )
        if gw is None:
            return torch.tensor(0.0)
        return load_balancing_loss(gw, self._n_experts)

    def get_group_consistency_loss(
        self, trait_group_indices: list[list[int]]
    ) -> torch.Tensor:
        gw = (
            self._gate_weights_grad
            if self._gate_weights_grad is not None
            else self.last_gate_weights
        )
        if gw is None or not trait_group_indices:
            return gw.new_zeros(()) if gw is not None else torch.tensor(0.0)
        total = gw.new_zeros(())
        n = 0
        for idx_list in trait_group_indices:
            if len(idx_list) < 2:
                continue
            g = gw[:, idx_list, :]
            mean_g = g.mean(dim=1, keepdim=True)
            var = ((g - mean_g) ** 2).mean()
            total = total + var
            n += 1
        return total / max(n, 1)
