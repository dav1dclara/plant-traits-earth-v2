from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_trait_ids(trait_dir: Path, n_traits: int) -> list[str]:
    """Resolve trait IDs from merged target rasters when available."""
    if trait_dir.exists():
        trait_ids = sorted(path.stem for path in trait_dir.glob("*.tif"))
        if len(trait_ids) >= n_traits:
            return trait_ids[:n_traits]
    return [f"trait_{idx:02d}" for idx in range(n_traits)]


def matrix_to_jsonable(matrix: np.ndarray) -> list[list[float | None]]:
    """Convert a matrix to nested Python lists with NaN values serialized as null."""
    matrix = np.asarray(matrix)
    return [
        [float(value) if np.isfinite(value) else None for value in row.tolist()]
        for row in matrix
    ]


def build_pair_records(
    matrix: np.ndarray,
    trait_ids: list[str],
) -> list[dict[str, float | str]]:
    """Build pairwise records for upper-triangular matrix values."""
    records = []
    n_traits = len(trait_ids)
    for i in range(n_traits):
        for j in range(i + 1, n_traits):
            score = matrix[i, j]
            if not np.isfinite(score):
                continue
            records.append(
                {
                    "trait_a": trait_ids[i],
                    "trait_b": trait_ids[j],
                    "score": float(score),
                }
            )
    return records


def compute_gate_similarity(mean_gate_weights: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between per-trait mean gate profiles."""
    mean_gate_weights = np.asarray(mean_gate_weights, dtype=np.float64)
    norms = np.linalg.norm(mean_gate_weights, axis=1, keepdims=True)
    safe = norms > 0
    normalized = np.zeros_like(mean_gate_weights)
    safe_rows = safe[:, 0]
    normalized[safe_rows] = mean_gate_weights[safe_rows] / norms[safe_rows]
    similarity = normalized @ normalized.T
    invalid = ~safe_rows
    similarity[invalid, :] = np.nan
    similarity[:, invalid] = np.nan
    return similarity.astype(np.float32)


def summarize_gate_weights(
    gate_weights: np.ndarray,
    trait_ids: list[str],
) -> dict:
    """Summarize per-trait gate usage for MMoE diagnostics."""
    gate_weights = np.asarray(gate_weights, dtype=np.float32)
    if gate_weights.ndim != 3:
        raise ValueError(
            f"Expected gate_weights shape (N, T, E), got {gate_weights.shape}"
        )

    n_samples, n_traits, n_experts = gate_weights.shape
    mean_gate_weights = gate_weights.mean(axis=0)
    std_gate_weights = gate_weights.std(axis=0)
    expert_usage_mean = gate_weights.mean(axis=(0, 1))
    expert_usage_by_trait = mean_gate_weights

    gate_entropy = -np.sum(
        gate_weights * np.log(np.clip(gate_weights, 1e-8, 1.0)),
        axis=-1,
    ) / np.log(float(n_experts))
    mean_entropy_per_trait = gate_entropy.mean(axis=0)
    mean_entropy = float(mean_entropy_per_trait.mean())

    dominant_expert_per_sample = np.argmax(gate_weights, axis=-1)
    dominant_counts = np.stack(
        [
            np.bincount(
                dominant_expert_per_sample[:, trait_idx],
                minlength=n_experts,
            )
            for trait_idx in range(n_traits)
        ],
        axis=0,
    )
    dominant_frequency = dominant_counts / max(n_samples, 1)
    dominant_expert = np.argmax(mean_gate_weights, axis=1)
    dominant_expert_frequency = dominant_frequency[np.arange(n_traits), dominant_expert]

    gate_similarity = compute_gate_similarity(mean_gate_weights)
    pair_records = build_pair_records(gate_similarity, trait_ids)
    pair_records_desc = sorted(
        pair_records, key=lambda record: record["score"], reverse=True
    )
    pair_records_asc = sorted(pair_records, key=lambda record: record["score"])

    per_trait = []
    for trait_idx, trait_id in enumerate(trait_ids):
        per_trait.append(
            {
                "trait_id": trait_id,
                "dominant_expert": int(dominant_expert[trait_idx]),
                "dominant_expert_weight": float(
                    mean_gate_weights[trait_idx, dominant_expert[trait_idx]]
                ),
                "dominant_expert_frequency": float(
                    dominant_expert_frequency[trait_idx]
                ),
                "mean_entropy": float(mean_entropy_per_trait[trait_idx]),
                "mean_gate_weights": [
                    float(value) for value in mean_gate_weights[trait_idx].tolist()
                ],
                "std_gate_weights": [
                    float(value) for value in std_gate_weights[trait_idx].tolist()
                ],
            }
        )

    expert_specialists = []
    for expert_idx in range(n_experts):
        top_indices = np.argsort(mean_gate_weights[:, expert_idx])[::-1][:10]
        expert_specialists.append(
            {
                "expert_index": int(expert_idx),
                "mean_usage": float(expert_usage_mean[expert_idx]),
                "top_traits": [trait_ids[idx] for idx in top_indices.tolist()],
                "top_trait_weights": [
                    float(mean_gate_weights[idx, expert_idx])
                    for idx in top_indices.tolist()
                ],
            }
        )

    return {
        "n_samples": int(n_samples),
        "n_traits": int(n_traits),
        "n_experts": int(n_experts),
        "mean_entropy": mean_entropy,
        "expert_usage_mean": [float(value) for value in expert_usage_mean.tolist()],
        "mean_gate_weights": matrix_to_jsonable(mean_gate_weights),
        "std_gate_weights": matrix_to_jsonable(std_gate_weights),
        "gate_similarity_matrix": matrix_to_jsonable(gate_similarity),
        "pairs": pair_records_desc,
        "top_positive_pairs": pair_records_desc[:25],
        "top_negative_pairs": pair_records_asc[:25],
        "per_trait": per_trait,
        "expert_specialists": expert_specialists,
        "dominant_expert_by_trait": [int(value) for value in dominant_expert.tolist()],
        "dominant_expert_frequency": [
            float(value) for value in dominant_expert_frequency.tolist()
        ],
        "_mean_gate_weights_array": mean_gate_weights,
        "_gate_similarity_array": gate_similarity,
    }
