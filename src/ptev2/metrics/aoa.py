from __future__ import annotations

import math

import torch
from tqdm import tqdm


def channelwise_patch_mean(x: torch.Tensor) -> torch.Tensor:
    """Per-sample channel mean over spatial dims, ignoring non-finite values."""
    finite = torch.isfinite(x)
    counts = finite.sum(dim=(2, 3))
    summed = torch.where(finite, x, torch.zeros_like(x)).sum(dim=(2, 3))
    return torch.where(
        counts > 0, summed / counts.clamp_min(1), torch.zeros_like(summed)
    )


def collect_patch_features(loader, device: torch.device) -> torch.Tensor:
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for X, _ in tqdm(loader, desc="Collecting AoA features"):
            X = X.to(device=device, dtype=torch.float32)
            features.append(channelwise_patch_mean(X).cpu())
    if not features:
        raise ValueError("No samples available to compute AoA features.")
    return torch.cat(features, dim=0)


def _min_distances_to_reference(
    queries: torch.Tensor,
    reference: torch.Tensor,
    block_size: int = 2048,
) -> torch.Tensor:
    mins = []
    for start in range(0, queries.shape[0], block_size):
        q_block = queries[start : start + block_size]
        dist = torch.cdist(q_block, reference)
        mins.append(dist.min(dim=1).values)
    return torch.cat(mins, dim=0)


def compute_aoa_metrics(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    q: float = 0.95,
) -> dict:
    train_mean = train_features.mean(dim=0)
    train_std = train_features.std(dim=0)
    train_std = torch.where(train_std > 1e-8, train_std, torch.ones_like(train_std))

    train_z = (train_features - train_mean) / train_std
    test_z = (test_features - train_mean) / train_std

    loo_min = []
    block_size = 1024
    n_train = train_z.shape[0]
    for start in range(0, n_train, block_size):
        stop = min(start + block_size, n_train)
        q_block = train_z[start:stop]
        dist = torch.cdist(q_block, train_z)
        row_idx = torch.arange(stop - start)
        col_idx = torch.arange(start, stop)
        dist[row_idx, col_idx] = math.inf
        loo_min.append(dist.min(dim=1).values)
    loo_min = torch.cat(loo_min, dim=0)

    aoa_threshold = float(torch.quantile(loo_min, q).item())
    test_min = _min_distances_to_reference(test_z, train_z)
    in_aoa = test_min <= aoa_threshold
    aoa_coverage = float(in_aoa.float().mean().item())

    return {
        "aoa_threshold": aoa_threshold,
        "aoa_quantile": q,
        "aoa_coverage": aoa_coverage,
        "aoa_outside_fraction": float(1.0 - aoa_coverage),
        "aoa_test_n": int(test_min.numel()),
    }
