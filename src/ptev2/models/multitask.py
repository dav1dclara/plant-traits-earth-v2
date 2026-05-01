"""
Fixed MTL/MMoE models restoring the original architecture.

Changes vs src/ptev2/models/multitask.py:
1. DenseMMoELayer: temperature scaling + top-k sparsification in gates
2. GatedMMoEModel: shared_projection + residual connection to task heads

v3 additions (GatedMMoEModelV3):
- Deeper task heads (3×3 conv like ResPatchCNN) to match STL prediction capacity
- Load-balancing auxiliary loss to prevent expert collapse
- Configurable residual scaling (default 0 = no residual shortcut)
- Higher default temperature and top-k for better expert utilization
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptev2.models.traitPatchCNN import ResidualBlock, _make_activation, _make_norm


class InputPCA(nn.Module):
    """Fixed per-pixel PCA projection as a 1x1 conv.

    Reduces C input channels to n_components using PCA eigenvectors fitted
    on training data. Acts as a learned-then-frozen whitening step that
    decorrelates collinear predictor channels (e.g. SoilGrids depth repeats,
    MODIS temporal autocorrelation) before the residual encoder.

    Weights are NOT trainable by default. Call init_from_pca() after fitting
    PCA on training data to set the projection matrix.
    """

    def __init__(self, in_channels: int, n_components: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, n_components, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.bias)
        for p in self.proj.parameters():
            p.requires_grad_(False)

    def init_from_pca(self, components: torch.Tensor, data_mean: torch.Tensor) -> None:
        """Initialise weights from fitted PCA.

        Args:
            components: (n_components, in_channels) top-K eigenvectors.
            data_mean:  (in_channels,) training-set channel mean.
        """
        with torch.no_grad():
            self.proj.weight.copy_(components.unsqueeze(-1).unsqueeze(-1))
            self.proj.bias.copy_(-(components @ data_mean))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SharedEncoder(nn.Module):
    """Shared residual encoder with optional PCA input projection."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "gn",
        activation: Literal["prelu", "silu", "gelu"] = "prelu",
        gn_groups: int = 8,
        stride_blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout_p: float = 0.0,
        pca_n_components: int | None = None,
    ) -> None:
        super().__init__()

        # Optional PCA projection: reduces in_channels → pca_n_components before encoder.
        self.input_pca: InputPCA | None = None
        effective_in = in_channels
        if pca_n_components is not None and 0 < pca_n_components < in_channels:
            self.input_pca = InputPCA(in_channels, pca_n_components)
            effective_in = pca_n_components

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.block1 = ResidualBlock(
            effective_in,
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
        if self.input_pca is not None:
            x = self.input_pca(x)
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
    """Single dense prediction head over the shared encoder."""

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
        pca_n_components: int | None = None,
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
            pca_n_components=pca_n_components,
        )
        self.head = TaskHead(self.encoder.out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class MTLModel(nn.Module):
    """Shared encoder with one prediction head per output channel.

    per_trait_conv=False (default / old behaviour):
        shared_head (one 3×3 conv for all traits) → per-trait 1×1 heads.
        All traits see the same intermediate features — effectively STL + split head.

    per_trait_conv=True (recommended for genuine MTL):
        No shared_head. Each trait gets its own 3×3 + 1×1 head so it can learn
        trait-specific spatial features from the encoder output.
        Costs ~37× more head params but ensures traits actually specialise.
    """

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
        pca_n_components: int | None = None,
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
            pca_n_components=pca_n_components,
        )
        c3 = self.encoder.out_channels
        self.per_trait_conv = per_trait_conv

        if per_trait_conv:
            # Each trait gets its own 3×3 → 1×1 head — genuine per-trait specialisation.
            # No shared_head: encoder output goes directly to each per-trait DeeperTaskHead.
            self.shared_head = None
            self.heads = nn.ModuleList(
                [
                    DeeperTaskHead(c3, 1, norm, activation, gn_groups, dropout_p)
                    for _ in range(out_channels)
                ]
            )
        else:
            # Legacy: shared 3×3 block before simple per-trait 1×1 heads.
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
            # Collect intermediate features (after 3×3 conv) for group-consistency loss.
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
        """Encourage traits in the same group to learn similar intermediate features.

        Minimizes variance of DeeperTaskHead intermediate features (features after
        the 3×3 conv+norm+PReLU+dropout, before the final 1×1 projection) within
        each group. Only active during training when per_trait_conv=True.

        Args:
            trait_group_indices: list of groups, each a list of 0-based trait indices.
        """
        if self._last_intermediates is None or not trait_group_indices:
            return torch.tensor(0.0)
        total = self._last_intermediates[0].new_zeros(())
        n = 0
        for idx_list in trait_group_indices:
            if len(idx_list) < 2:
                continue
            # Stack group's intermediate features: (B, k, C, H, W)
            g = torch.stack([self._last_intermediates[i] for i in idx_list], dim=1)
            mean_g = g.mean(dim=1, keepdim=True)
            var = ((g - mean_g) ** 2).mean()
            total = total + var
            n += 1
        return total / max(n, 1)


class DenseMMoELayer(nn.Module):
    """
    Dense Mixture-of-Experts with temperature scaling and optional top-k gating.

    Differences from merged multitask.py:
    - Gates produce raw logits (no baked-in Softmax)
    - Temperature scaling applied before softmax
    - Optional top-k sparsification
    """

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

        # Gates produce raw logits — softmax applied with temperature in forward
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
        """Apply temperature scaling and optional top-k sparsification."""
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


class GatedMMoEModel(nn.Module):
    """
    Dense MMoE model with shared projection and residual connection.

    Differences from merged MMoEModel:
    1. shared_projection: projects encoder output to expert_hidden dim
    2. Residual connection: task_features = routed_map + shared_map
    3. Temperature and top-k gating via DenseMMoELayer
    """

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
        pca_n_components: int | None = None,
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
            pca_n_components=pca_n_components,
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

        # Shared projection: encoder features → expert_hidden dim (MISSING in merged code)
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
        shared_map = self.shared_projection(z)  # (B, expert_hidden, H, W)
        self.last_gate_weights = gate_weights.detach()

        outputs = []
        for i, head in enumerate(self.heads):
            # Residual connection: routed expert features + shared encoder features
            task_features = task_maps[:, i] + shared_map
            outputs.append(head(task_features))

        return torch.cat(outputs, dim=1)


# ---------------------------------------------------------------------------
# V3: Improved MMoE addressing the issues found in ANALYSIS_v2.md
# ---------------------------------------------------------------------------


class DeeperTaskHead(nn.Module):
    """Task head matching ResPatchCNN's deeper head (3×3 conv + 1×1 conv).

    This closes the prediction-capacity gap between STL and MMoE/MTL.
    """

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
        """Forward returning (output, intermediate).

        intermediate: features after the 3×3 conv+norm+PReLU+dropout,
        before the final 1×1 projection — used for group-consistency loss
        in the tuned MTLModel.
        """
        inter = self.head[:-1](x)  # layers 0-3: 3×3 conv + norm + PReLU + dropout
        return self.head[-1](inter), inter


def load_balancing_loss(gate_weights: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Compute load-balancing auxiliary loss (Switch Transformer style).

    Encourages uniform expert utilization across the batch.

    Args:
        gate_weights: (B, n_tasks, n_experts) — soft or sparse gate weights.
        n_experts: number of experts.

    Returns:
        Scalar loss, 0 when perfectly balanced.
    """
    # fraction of tokens routed to each expert, averaged over tasks
    # gate_weights: (B, T, E)
    f = gate_weights.mean(dim=(0, 1))  # (E,) — mean gate weight per expert
    # ideal uniform: 1/n_experts for each
    target = torch.full_like(f, 1.0 / n_experts)
    # CV-squared style: variance of load / mean^2
    return n_experts * (f * f).sum()  # minimized when f is uniform


class GatedMMoEModelV3(nn.Module):
    """Improved dense MMoE model fixing key issues from v2.

    Key changes vs GatedMMoEModel:
    1. Deeper task heads (3×3 + 1×1 conv) matching ResPatchCNN capacity
    2. No residual shortcut by default (residual_scale=0.0)
    3. Higher default temperature (1.0) and top_k (4) for better expert usage
    4. Exposes load_balancing_loss() for training loop integration
    """

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
        pca_n_components: int | None = None,
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
            pca_n_components=pca_n_components,
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

        # Shared projection for optional residual (only used if residual_scale > 0)
        if residual_scale > 0:
            self.shared_projection = nn.Sequential(
                nn.Conv2d(c3, expert_hidden, kernel_size=1, bias=False),
                _make_norm(norm, expert_hidden, gn_groups),
                _make_activation(activation, expert_hidden),
            )
        else:
            self.shared_projection = None

        # Deeper task heads — matches ResPatchCNN head capacity
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
        self._gate_weights_grad: torch.Tensor | None = (
            None  # non-detached, for aux losses
        )
        self._n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        route_features = self.route_pool(z).flatten(1)
        task_maps, gate_weights = self.mmoe(z, route_features)
        self.last_gate_weights = gate_weights.detach()  # for analysis/logging
        self._gate_weights_grad = (
            gate_weights  # for loss computation (has grad during train)
        )

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
        """Load-balancing loss — uses non-detached gate weights during training
        so gradients flow back through the gate MLPs."""
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
        """Encourage traits in the same group to route to similar experts.

        For each group, minimizes variance of gate distributions within the group.
        This uses non-detached gate weights so gradients flow to gate MLPs.

        Args:
            trait_group_indices: list of groups, each group is a list of
                0-based channel indices for traits in that group.
        """
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
            g = gw[:, idx_list, :]  # (B, k, n_experts)
            mean_g = g.mean(dim=1, keepdim=True)  # (B, 1, n_experts)
            var = ((g - mean_g) ** 2).mean()
            total = total + var
            n += 1
        return total / max(n, 1)


# ---------------------------------------------------------------------------
# Improved loss: UncertaintyWeightedMTLLossV2 with configurable error type
# ---------------------------------------------------------------------------


def _per_trait_masked_loss_v2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
    w_gbif: float = 1.0,
    w_splot: float = 2.0,
    error_type: str = "mse",
    huber_delta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-output masked loss with configurable error type (MSE or smooth_l1)."""
    if prediction.ndim == 4:
        prediction = prediction.permute(0, 2, 3, 1).reshape(-1, prediction.shape[1])
        target = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
        source_mask = source_mask.permute(0, 2, 3, 1).reshape(-1, source_mask.shape[1])

    weights = torch.zeros_like(target, dtype=prediction.dtype)
    weights = torch.where(source_mask == 1, torch.full_like(weights, w_gbif), weights)
    weights = torch.where(source_mask == 2, torch.full_like(weights, w_splot), weights)

    valid = torch.isfinite(prediction) & torch.isfinite(target) & (weights > 0)
    valid_f = valid.to(dtype=prediction.dtype)

    pred_safe = torch.where(valid, prediction, torch.zeros_like(prediction))
    tgt_safe = torch.where(valid, target, torch.zeros_like(target))

    if error_type == "mse":
        error = (pred_safe - tgt_safe).pow(2)
    elif error_type in ("smooth_l1", "huber"):
        error = F.smooth_l1_loss(
            pred_safe, tgt_safe, reduction="none", beta=huber_delta
        )
    else:
        raise ValueError(f"Unsupported error_type: {error_type}")

    numerator = (error * weights).sum(dim=0)
    denominator = (weights * valid_f).sum(dim=0).clamp_min(eps)
    trait_losses = numerator / denominator

    no_valid = valid_f.sum(dim=0) == 0
    if bool(no_valid.any()):
        trait_losses = trait_losses.clone()
        trait_losses[no_valid] = float("nan")
    return trait_losses


class UncertaintyWeightedMTLLossV2(nn.Module):
    """Uncertainty-weighted MTL loss with configurable base error.

    Unlike the original (ptev2.loss.UncertaintyWeightedMTLLoss) which always
    uses MSE, this version supports smooth_l1 for a fair comparison with
    WeightedMaskedDenseLoss.
    """

    def __init__(
        self,
        n_traits: int = 37,
        w_gbif: float = 1.0,
        w_splot: float = 2.0,
        init_log_sigma_sq: float = 0.0,
        error_type: str = "smooth_l1",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.log_sigma_sq = nn.Parameter(
            torch.full((n_traits,), float(init_log_sigma_sq), dtype=torch.float32)
        )
        self.w_gbif = float(w_gbif)
        self.w_splot = float(w_splot)
        self.error_type = str(error_type)
        self.huber_delta = float(huber_delta)

    def loss_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trait_losses = _per_trait_masked_loss_v2(
            prediction,
            target,
            source_mask,
            w_gbif=self.w_gbif,
            w_splot=self.w_splot,
            error_type=self.error_type,
            huber_delta=self.huber_delta,
        )

        valid = torch.isfinite(trait_losses)
        if not bool(valid.any()):
            zero = prediction.sum() * 0.0
            return zero, zero

        trait_losses = torch.nan_to_num(trait_losses, nan=0.0)
        weighted = (
            torch.exp(-self.log_sigma_sq[valid]) * trait_losses[valid]
            + self.log_sigma_sq[valid]
        )
        numerator = weighted.sum()
        denominator = valid.sum().to(dtype=prediction.dtype)
        return numerator, denominator

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        numerator, denominator = self.loss_components(prediction, target, source_mask)
        if not bool((denominator > 0).item()):
            return prediction.sum() * 0.0
        return numerator / denominator


# ---------------------------------------------------------------------------
# GradNorm-style dynamic per-task loss weighting
# ---------------------------------------------------------------------------


class GradNormMTLLoss(nn.Module):
    """Dynamic per-task loss weighting inspired by GradNorm (Chen et al., 2018).

    Tracks an EMA of per-task losses to estimate relative training rates.
    Tasks improving slower than the average get higher loss weight, which
    encourages balanced learning across all 37 traits.

    Simplified vs. full GradNorm: uses loss-ratio tracking instead of computing
    per-task gradient norms w.r.t. the shared encoder (avoids expensive
    second-order passes while preserving the core adaptive-weighting insight).

    Interface matches WeightedMaskedDenseLoss (loss_components returns
    numerator/denominator), so it works as a drop-in in trainv2.py.
    """

    def __init__(
        self,
        n_traits: int = 37,
        alpha: float = 1.5,
        ema_decay: float = 0.98,
        gradnorm_weight: float = 0.1,
        w_gbif: float = 1.0,
        w_splot: float = 16.0,
        error_type: str = "smooth_l1",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_traits = int(n_traits)
        self.alpha = float(alpha)
        self.ema_decay = float(ema_decay)
        self.gradnorm_weight = float(gradnorm_weight)
        self.w_gbif = float(w_gbif)
        self.w_splot = float(w_splot)
        self.error_type = str(error_type)
        self.huber_delta = float(huber_delta)

        # Learnable log-weights per task (included in optimizer via loss_fn.parameters())
        self.log_weights = nn.Parameter(torch.zeros(self.n_traits))

        # EMA buffers — no gradient, updated with torch.no_grad()
        self.register_buffer("ema_losses", torch.ones(self.n_traits))
        self.register_buffer("initial_losses", torch.zeros(self.n_traits))
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def weights(self) -> torch.Tensor:
        """Normalized weights summing to n_traits (uniform = all ones)."""
        w = self.log_weights.exp()
        return w / w.sum() * self.n_traits

    def _update_ema(self, per_task_losses: torch.Tensor) -> None:
        """Update EMA of per-task losses (no gradient)."""
        with torch.no_grad():
            valid = torch.isfinite(per_task_losses)
            losses = (
                per_task_losses.detach()
                .where(valid, torch.ones_like(per_task_losses))
                .clamp_min(1e-8)
            )
            if not bool(self.initialized.item()):
                self.initial_losses.copy_(losses)
                self.ema_losses.copy_(losses)
                self.initialized.fill_(True)
            else:
                self.ema_losses = torch.where(
                    valid,
                    self.ema_decay * self.ema_losses + (1 - self.ema_decay) * losses,
                    self.ema_losses,
                )

    def _target_weights(self) -> torch.Tensor:
        """Target weights: tasks improving slower get higher weight."""
        if not bool(self.initialized.item()):
            return torch.ones(self.n_traits, device=self.log_weights.device)
        L_hat = self.ema_losses / (self.initial_losses + 1e-8)  # relative training rate
        r = L_hat / (L_hat.mean() + 1e-8)
        target = r.pow(self.alpha)
        return target / target.sum() * self.n_traits

    def loss_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (numerator, denominator) matching WeightedMaskedDenseLoss."""
        per_task_losses = _per_trait_masked_loss_v2(
            prediction,
            target,
            source_mask,
            w_gbif=self.w_gbif,
            w_splot=self.w_splot,
            error_type=self.error_type,
            huber_delta=self.huber_delta,
        )

        # Update EMA (no grad)
        self._update_ema(per_task_losses)

        valid = torch.isfinite(per_task_losses)
        if not bool(valid.any()):
            zero = prediction.sum() * 0.0
            return zero, zero

        weights = self.weights
        weighted_losses = weights[valid] * per_task_losses[valid]

        # GradNorm regularization: drive learned weights toward target
        target_w = self._target_weights().detach()
        gnorm_reg = ((weights - target_w) ** 2).mean()

        numerator = weighted_losses.sum() + self.gradnorm_weight * gnorm_reg
        denominator = valid.sum().to(dtype=prediction.dtype)
        return numerator, denominator

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        num, den = self.loss_components(prediction, target, source_mask)
        if not bool((den > 0).item()):
            return prediction.sum() * 0.0
        return num / den


# ---------------------------------------------------------------------------
# Auxiliary quantile prediction head (q05 / q95 supervision)
# ---------------------------------------------------------------------------


class AuxQuantileHead(nn.Module):
    """Lightweight per-trait quantile predictor branching off model output.

    Takes the model's primary mean predictions (B, T, H, W) and learns to
    predict q05 and q95 for each trait via a depthwise 1×1 convolution.
    Output shape: (B, 2*T, H, W) — first T channels are q05, next T are q95.

    Kept separate from the main model so it is NOT saved in the primary
    checkpoint and does not affect test.py evaluation.  Gradients flow back
    through mean_pred into the shared backbone, providing extra supervision.

    Initialization: q05 ≈ mean − 1.0, q95 ≈ mean + 1.0 (rough z-score priors).
    """

    def __init__(self, n_traits: int) -> None:
        super().__init__()
        self.n_traits = int(n_traits)
        # groups=n_traits → independent (scale, shift) per trait for each quantile
        self.head = nn.Conv2d(
            n_traits, n_traits * 2, kernel_size=1, groups=n_traits, bias=True
        )
        # Prior: q05 ≈ mean - 1, q95 ≈ mean + 1 in z-score space
        with torch.no_grad():
            self.head.weight.fill_(1.0)
            self.head.bias[:n_traits].fill_(-1.0)  # q05 offset
            self.head.bias[n_traits:].fill_(1.0)  # q95 offset

    def forward(self, mean_pred: torch.Tensor) -> torch.Tensor:
        """Returns (B, 2*T, H, W): [q05_t0..q05_tN, q95_t0..q95_tN]."""
        return self.head(mean_pred)
