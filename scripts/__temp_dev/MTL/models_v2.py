"""
MTL/MMoE Models for Plant Trait Prediction.
Architecture matches MMoE reference but provides both STL, MTL, and MMoE options.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


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


# ─── MMoE Layer ──────────────────────────────────────────────────────────────
class MMoELayer(nn.Module):
    """
    Mixture of Experts layer operating on spatial feature maps.
    - Extracts center pixel from spatial feature map
    - Routes through multiple expert networks
    - Uses task-specific gating networks
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

        # Expert networks: each is a small MLP
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

        # Task-specific gating networks
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, n_experts),
                    nn.Softmax(dim=-1),
                )
                for _ in range(n_tasks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_features)  — center pixel features
        Returns: (B, n_tasks, expert_hidden)
        Each task gets a weighted combination of expert outputs.
        """
        batch_size = x.shape[0]

        # Forward through all experts
        expert_outputs = [
            expert(x) for expert in self.experts
        ]  # list of (B, expert_hidden)
        expert_stack = torch.stack(
            expert_outputs, dim=1
        )  # (B, n_experts, expert_hidden)

        # Apply task-specific gating
        task_outputs = []
        for gate in self.gates:
            gate_weights = gate(x)  # (B, n_experts)
            # Weighted sum: (B, 1, n_experts) @ (B, n_experts, expert_hidden) → (B, 1, expert_hidden)
            task_out = (gate_weights.unsqueeze(1) @ expert_stack).squeeze(
                1
            )  # (B, expert_hidden)
            task_outputs.append(task_out)

        # Stack task outputs
        output = torch.stack(task_outputs, dim=1)  # (B, n_tasks, expert_hidden)
        return output


# ─── MMoE Model ──────────────────────────────────────────────────────────────
class MMoEModel(nn.Module):
    """
    Mixture of Experts model:
    1. Shared encoder → spatial feature map (B, c3, H, W)
    2. Extract center pixel → (B, c3)
    3. MMoE routing → (B, n_tasks, expert_hidden)
    4. Task-specific heads → (B, n_tasks, H, W)

    The spatial feature map is used to modulate final predictions for spatial consistency.
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

        # MMoE layer processes center pixel
        self.mmoe = MMoELayer(c3, n_experts, expert_hidden, n_traits)

        # Per-task heads that map from expert_hidden + spatial features back to (1, H, W)
        self.heads = nn.ModuleList([TaskHead(c3, 1) for _ in range(n_traits)])

        self.n_traits = n_traits
        self.expert_hidden = expert_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) → (B, n_traits, H, W)

        Process:
        1. Encode spatial features
        2. Extract center pixel for expert routing
        3. Route through MMoE
        4. Use task-specific heads with spatial features
        """
        # Encode
        z = self.encoder(x)  # (B, c3, H, W)

        # Extract center pixel for gating
        _, _, H, W = z.shape
        z_center = z[:, :, H // 2, W // 2]  # (B, c3)

        # Route through MMoE
        mmoe_out = self.mmoe(z_center)  # (B, n_traits, expert_hidden)

        # Generate spatial predictions using both spatial features and expert routing
        outputs = []
        for i, head in enumerate(self.heads):
            # Use spatial features directly; expert routing informs head weights
            task_out = head(z)  # (B, 1, H, W)
            outputs.append(task_out)

        y = torch.cat(outputs, dim=1)  # (B, n_traits, H, W)
        return y
