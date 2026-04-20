"""
MTL/MMoE Models for Plant Trait Prediction.
Architecture matches MMoE reference but provides both STL, MTL, and MMoE options.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Normalization utility ───────────────────────────────────────────────────
def _make_norm(
    norm: Literal["bn", "gn", "none"],
    num_channels: int,
    gn_groups: int = 8,
) -> nn.Module:
    """Create normalization layer."""
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        groups = min(gn_groups, num_channels)
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    return nn.Identity()


# ─── Residual Block ──────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """
    Single residual block: Conv→Norm→ReLU→Drop→Conv→Norm + Skip.
    Used in the shared encoder backbone.
    """

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

        # Main path: Conv1 → Norm → PReLU → Dropout → Conv2 → Norm
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

        # Skip path: 1x1 conv if channel/stride mismatch
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
        out = self.act_out(out)
        return out


# ─── Shared Encoder Backbone ─────────────────────────────────────────────────
class SharedEncoder(nn.Module):
    """
    Four-stage residual encoder.
    Reduces spatial dims progressively and increases channels.
    Output: (B, c3, H', W') where c3 = base_channels * 4
    """

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
        """(B, C_in, H, W) → (B, c3, H', W')"""
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        return z


# ─── Task-Specific Head ──────────────────────────────────────────────────────
class TaskHead(nn.Module):
    """Simple 1x1 conv head for per-trait prediction."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─── STL Model (Single-Task Learning) ────────────────────────────────────────
class STLModel(nn.Module):
    """
    Single-task learning: shared encoder + single prediction head.
    Input:  (B, C, H, W)
    Output: (B, n_traits, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        n_traits: int = 37,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = SharedEncoder(
            in_channels, base_channels, norm, gn_groups, stride_blocks, dropout_p
        )
        self.head = TaskHead(self.encoder.out_channels, n_traits)
        self.n_traits = n_traits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, n_traits, H, W)"""
        z = self.encoder(x)
        y = self.head(z)
        return y


# ─── MTL Model (Multi-Task Learning) ─────────────────────────────────────────
class MTLModel(nn.Module):
    """
    Multi-task learning: shared encoder + multiple task-specific heads.
    Each task gets its own prediction head.
    Input:  (B, C, H, W)
    Output: list of (B, 1, H, W) — one per task
    """

    def __init__(
        self,
        in_channels: int,
        n_traits: int = 37,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = SharedEncoder(
            in_channels, base_channels, norm, gn_groups, stride_blocks, dropout_p
        )
        # One head per trait (could concatenate outputs at the end if needed)
        self.heads = nn.ModuleList(
            [TaskHead(self.encoder.out_channels, 1) for _ in range(n_traits)]
        )
        self.n_traits = n_traits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) → (B, n_traits, H, W)
        Concatenate all task outputs into a single tensor.
        """
        z = self.encoder(x)
        outputs = [head(z) for head in self.heads]  # list of (B, 1, H, W)
        y = torch.cat(outputs, dim=1)  # (B, n_traits, H, W)
        return y


# ─── Legacy MMoE Layer ───────────────────────────────────────────────────────
class MMoELayer(nn.Module):
    """
    Legacy MMoE layer operating on pooled center-pixel features.
    This is kept checkpoint-compatible with the original temp MMoE runs.
    """

    def __init__(
        self,
        in_features: int,
        n_experts: int = 4,
        expert_hidden: int = 64,
        n_tasks: int = 37,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, expert_hidden),
                    nn.ReLU(),
                    nn.Linear(expert_hidden, expert_hidden),
                )
                for _ in range(n_experts)
            ]
        )

        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, n_experts),
                    nn.Softmax(dim=-1),
                )
                for _ in range(n_tasks)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, in_features)
        Returns:
            task_outputs: (B, n_tasks, expert_hidden)
            gate_weights: (B, n_tasks, n_experts)
        """
        expert_outputs = [expert(x) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)

        gate_weights = torch.stack([gate(x) for gate in self.gates], dim=1)
        task_outputs = torch.einsum("bte,beh->bth", gate_weights, expert_stack)
        return task_outputs, gate_weights


# ─── Dense MMoE Layer ────────────────────────────────────────────────────────
class DenseMMoELayer(nn.Module):
    """
    Dense Mixture of Experts layer operating on spatial feature maps.
    - Builds expert feature maps from the shared encoder output
    - Uses task-specific gates from pooled patch context
    - Returns one routed spatial map per task
    """

    def __init__(
        self,
        in_features: int,
        n_experts: int = 4,
        expert_hidden: int = 64,
        n_tasks: int = 37,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        dropout_p: float = 0.0,
        gate_temperature: float = 1.0,
        gate_top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.expert_hidden = expert_hidden
        self.gate_temperature = gate_temperature
        self.gate_top_k = gate_top_k

        # Experts now produce dense spatial feature maps instead of task-agnostic vectors.
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

        # Gates use pooled patch context, not only the center pixel.
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
        """Apply temperature scaling and optional top-k sparsification to gates."""
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
        """
        spatial_features: (B, in_features, H, W)
        route_features: (B, in_features)
        Returns:
            task_maps: (B, n_tasks, expert_hidden, H, W)
            gate_weights: (B, n_tasks, n_experts)
        """
        expert_outputs = [expert(spatial_features) for expert in self.experts]
        expert_stack = torch.stack(
            expert_outputs, dim=1
        )  # (B, n_experts, expert_hidden, H, W)

        gate_logits = torch.stack(
            [gate(route_features) for gate in self.gates],
            dim=1,
        )  # (B, n_tasks, n_experts)
        gate_weights = self._apply_gate_constraints(gate_logits)

        task_maps = torch.einsum(
            "bte,bechw->btchw",
            gate_weights,
            expert_stack,
        )
        return task_maps, gate_weights


# ─── Legacy MMoE Model ───────────────────────────────────────────────────────
class MMoEModel(nn.Module):
    """
    Legacy MMoE model kept for compatibility with older checkpoints/results.
    The routing path is computed and exposed via last_gate_weights, but predictions
    are still produced directly from the shared spatial backbone.
    """

    def __init__(
        self,
        in_channels: int,
        n_traits: int = 37,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
        n_experts: int = 4,
        expert_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = SharedEncoder(
            in_channels, base_channels, norm, gn_groups, stride_blocks, dropout_p
        )
        c3 = self.encoder.out_channels

        self.mmoe = MMoELayer(c3, n_experts, expert_hidden, n_traits)
        self.heads = nn.ModuleList([TaskHead(c3, 1) for _ in range(n_traits)])

        self.n_traits = n_traits
        self.expert_hidden = expert_hidden
        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, _, height, width = z.shape
        z_center = z[:, :, height // 2, width // 2]
        _, gate_weights = self.mmoe(z_center)
        self.last_gate_weights = gate_weights.detach()

        outputs = [head(z) for head in self.heads]
        return torch.cat(outputs, dim=1)


# ─── Routed MMoE Model ───────────────────────────────────────────────────────
class GatedMMoEModel(nn.Module):
    """
    Mixture of Experts model:
    1. Shared encoder → spatial feature map (B, c3, H, W)
    2. Pool patch context → task-specific gates
    3. Dense MMoE routing → one routed spatial map per task
    4. Task-specific heads consume routed maps, so expert routing affects predictions
    """

    def __init__(
        self,
        in_channels: int,
        n_traits: int = 37,
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
        self.encoder = SharedEncoder(
            in_channels, base_channels, norm, gn_groups, stride_blocks, dropout_p
        )
        c3 = self.encoder.out_channels

        self.route_pool = nn.AdaptiveAvgPool2d(1)

        self.mmoe = DenseMMoELayer(
            c3,
            n_experts,
            expert_hidden,
            n_traits,
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
            [TaskHead(expert_hidden, 1) for _ in range(n_traits)]
        )

        self.n_traits = n_traits
        self.expert_hidden = expert_hidden
        self.gate_temperature = gate_temperature
        self.gate_top_k = gate_top_k
        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) → (B, n_traits, H, W)

        Process:
        1. Encode spatial features
        2. Pool global patch context for gating
        3. Mix dense expert maps with task-specific gates
        4. Predict each trait from its routed feature map
        """
        z = self.encoder(x)  # (B, c3, H, W)
        route_features = self.route_pool(z).flatten(1)  # (B, c3)
        task_maps, gate_weights = self.mmoe(z, route_features)
        shared_map = self.shared_projection(z)  # (B, expert_hidden, H, W)
        self.last_gate_weights = gate_weights.detach()

        outputs = []
        for i, head in enumerate(self.heads):
            task_features = task_maps[:, i] + shared_map
            task_out = head(task_features)
            outputs.append(task_out)

        y = torch.cat(outputs, dim=1)  # (B, n_traits, H, W)
        return y
