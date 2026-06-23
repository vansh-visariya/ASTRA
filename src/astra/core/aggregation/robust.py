"""
Robust Aggregation Implementations.

Byzantine-resilient aggregation: trimmed mean, coordinate-wise median,
and a hybrid pipeline with norm clipping, anomaly/similarity filtering,
and trust + staleness weighting.
"""

from typing import Any

import numpy as np


def trimmed_mean(updates: list[np.ndarray], trim_ratio: float) -> np.ndarray:
    if not updates:
        return np.array([])
    if len(updates) == 1:
        return updates[0].copy()

    arr = np.nan_to_num(np.array(updates), nan=0.0, posinf=1e6, neginf=-1e6)
    n_clients, dim = arr.shape
    trim_count = int(n_clients * trim_ratio)
    if trim_count == 0:
        return np.mean(arr, axis=0)

    sorted_indices = np.argsort(arr, axis=0)
    trimmed = arr.copy()
    for d in range(dim):
        mask = np.ones(n_clients, dtype=bool)
        mask[sorted_indices[:trim_count, d]] = False
        mask[sorted_indices[-trim_count:, d]] = False
        trimmed[:, d] = np.where(mask, arr[:, d], np.nan)

    return np.nan_to_num(np.nanmean(trimmed, axis=0), nan=0.0)


def coordinate_median(updates: list[np.ndarray]) -> np.ndarray:
    if not updates:
        return np.array([])
    if len(updates) == 1:
        return updates[0].copy()
    return np.nanmedian(np.array(updates), axis=0)


def _clip_norms(updates: list[np.ndarray], norm_clip: float) -> list[np.ndarray]:
    """Clip each update to norm_clip and return (clipped_updates, norms)."""
    norms = []
    clipped = []
    for u in updates:
        n = np.linalg.norm(u)
        norms.append(n)
        clipped.append(u / n * norm_clip if n > norm_clip else u)
    return clipped, norms


def _filter_anomalies(
    clipped: list[np.ndarray], norms: list[float], anomaly_k: float
) -> tuple[list[np.ndarray], list[int]]:
    """Remove norm-based outliers. Returns (filtered_updates, surviving_indices)."""
    norms_arr = np.array(norms)
    mean, std = np.mean(norms_arr), np.std(norms_arr)
    threshold = mean + anomaly_k * std if std > 0 else float("inf")
    indices = [i for i, n in enumerate(norms) if n <= threshold]
    filtered = [clipped[i] for i in indices]
    if not filtered:
        return [clipped[0]], [0]
    return filtered, indices


def _filter_by_similarity(
    updates: list[np.ndarray], sim_threshold: float
) -> tuple[list[np.ndarray], list[int]]:
    """Keep updates similar to the coordinate-wise median."""
    baseline = coordinate_median(updates)
    baseline_norm = np.linalg.norm(baseline)
    scores = []
    for u in updates:
        if baseline_norm > 1e-8:
            cos = np.dot(u, baseline) / (np.linalg.norm(u) * baseline_norm)
        else:
            cos = 1.0
        scores.append(max(cos, 0))
    indices = [i for i, s in enumerate(scores) if s >= sim_threshold]
    filtered = [updates[i] for i in indices]
    if not filtered:
        return [updates[0]], [0]
    return filtered, indices


def _weighted_sum(
    updates: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    """Compute weighted sum of updates."""
    result = np.zeros_like(updates[0])
    for i, u in enumerate(updates):
        result += weights[i] * np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
    return result


def hybrid_aggregator(
    updates: list[np.ndarray],
    trust_scores: list[float],
    staleness_weights: list[float],
    config: dict[str, Any],
    dataset_sizes: list[int] | None = None,
) -> np.ndarray:
    if not updates:
        return np.array([])
    if len(updates) == 1:
        return updates[0].copy()

    if dataset_sizes is None:
        dataset_sizes = [1] * len(updates)

    rc = config.get("robust", {})
    norm_clip = rc.get("norm_clip", 5.0)
    anomaly_k = rc.get("anomaly_k", 3.0)
    sim_threshold = rc.get("sim_threshold", 0.2)
    trim_ratio = rc.get("trim_ratio", 0.1)
    trust_power = rc.get("trust_power", 1.0)

    # Stage 1: norm clipping
    clipped, norms = _clip_norms(updates, norm_clip)

    # Stage 2: anomaly filtering
    filtered, indices = _filter_anomalies(clipped, norms, anomaly_k)
    filtered_trust = [trust_scores[i] for i in indices]

    # Stage 3: similarity filtering
    filtered, sim_indices = _filter_by_similarity(filtered, sim_threshold)
    final_trust = [filtered_trust[i] for i in sim_indices]
    final_staleness = [staleness_weights[indices[i]] for i in sim_indices]
    final_sizes = [dataset_sizes[indices[i]] for i in sim_indices]

    # Stage 4: robust aggregation
    robust_delta = trimmed_mean(filtered, trim_ratio)
    robust_delta = np.nan_to_num(robust_delta, nan=0.0, posinf=1e6, neginf=-1e6)

    # Stage 5: trust + staleness + data weighting
    weights = np.array(final_trust) ** trust_power * np.array(final_staleness)
    if final_sizes:
        data_w = np.array(final_sizes) / sum(final_sizes)
        weights *= data_w
    weights = np.nan_to_num(weights, nan=0.0)
    weights = weights / (np.sum(weights) + 1e-8)

    return _weighted_sum(filtered, weights)
