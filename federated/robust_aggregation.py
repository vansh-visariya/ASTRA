"""
Robust Aggregation Implementations.

Implements Byzantine-resilient aggregation methods including:
- Coordinate-wise median
- Trimmed mean
- Hybrid robust pipeline with trust scoring

References:
- Yin et al., "Byzantine-Robust Distributed Learning"
- Chen et al., "Distributed Learning with Heterogeneous Data"
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


def trimmed_mean(updates: List[np.ndarray], trim_ratio: float) -> np.ndarray:
    """
    Compute coordinate-wise trimmed mean.
    
    For each coordinate, remove trim_ratio fraction from both ends
    and compute mean of remaining values.
    
    Args:
        updates: List of client update vectors.
        trim_ratio: Fraction to trim from each end (0 to 0.5).
    
    Returns:
        Aggregated vector.
    """
    if not updates:
        return np.array([])
    
    if len(updates) == 1:
        return updates[0].copy()
    
    updates_array = np.array(updates)
    
    updates_array = np.nan_to_num(updates_array, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n_clients, dim = updates_array.shape
    
    trim_count = int(n_clients * trim_ratio)
    
    if trim_count == 0:
        return np.mean(updates_array, axis=0)
    
    sorted_indices = np.argsort(updates_array, axis=0)
    
    trimmed = updates_array.copy()
    for d in range(dim):
        lower_indices = sorted_indices[:trim_count, d]
        upper_indices = sorted_indices[-(trim_count):, d]
        
        mask = np.ones(n_clients, dtype=bool)
        mask[lower_indices] = False
        mask[upper_indices] = False
        
        trimmed[:, d] = np.where(mask, updates_array[:, d], np.nan)
    
    result = np.nanmean(trimmed, axis=0)
    
    result = np.nan_to_num(result, nan=0.0)
    
    return result


def coordinate_median(updates: List[np.ndarray]) -> np.ndarray:
    """
    Compute coordinate-wise median.
    
    Args:
        updates: List of client update vectors.
    
    Returns:
        Aggregated vector using coordinate-wise median.
    """
    if not updates:
        return np.array([])
    
    if len(updates) == 1:
        return updates[0].copy()
    
    updates_array = np.array(updates)
    
    result = np.median(updates_array, axis=0)
    
    return result


def compute_trust_weights(trust_scores: List[float], trust_power: float) -> np.ndarray:
    """Compute normalized trust weights."""
    if not trust_scores:
        return np.array([])
    
    weights = np.array(trust_scores) ** trust_power
    weights = weights / np.sum(weights)
    
    return weights


def compute_staleness_weights(staleness_values: List[float], async_lambda: float) -> np.ndarray:
    """Compute staleness-based weights using exponential decay."""
    if not staleness_values:
        return np.array([])
    
    weights = np.exp(-async_lambda * np.array(staleness_values))
    weights = weights / np.sum(weights)
    
    return weights


def hybrid_aggregator(
    updates: List[np.ndarray],
    trust_scores: List[float],
    staleness_weights: List[float],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Hybrid robust aggregator with multiple filtering stages.
    
    Implements:
    1. L2 norm clipping
    2. Anomaly detection (norm-based)
    3. Similarity filtering
    4. Robust operator (trimmed mean or median)
    5. Trust + staleness weighting
    
    Args:
        updates: List of client update vectors.
        trust_scores: List of trust scores for each client.
        staleness_weights: List of staleness weights.
        config: Configuration dictionary.
    
    Returns:
        Final aggregated delta.
    """
    if not updates:
        return np.array([])
    
    if len(updates) == 1:
        return updates[0].copy()
    
    robust_config = config.get('robust', {})
    trust_config = config.get('trust', {})
    
    norm_clip = robust_config.get('norm_clip', 5.0)
    anomaly_k = robust_config.get('anomaly_k', 3.0)
    sim_threshold = robust_config.get('sim_threshold', 0.2)
    trim_ratio = robust_config.get('trim_ratio', 0.1)
    trust_power = robust_config.get('trust_power', 1.0)
    
    n_clients = len(updates)
    
    clipped_updates = []
    norms = []
    for update in updates:
        norm = np.linalg.norm(update)
        norms.append(norm)
        
        if norm > norm_clip:
            clipped = (update / norm) * norm_clip
        else:
            clipped = update
        
        clipped_updates.append(clipped)
    
    norms = np.array(norms)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    suspicious_mask = np.ones(n_clients, dtype=bool)
    if std_norm > 0:
        threshold = mean_norm + anomaly_k * std_norm
        suspicious_mask = norms <= threshold
    
    filtered_updates = [clipped_updates[i] for i in range(n_clients) if suspicious_mask[i]]
    filtered_trust = [trust_scores[i] for i in range(n_clients) if suspicious_mask[i]]
    
    if not filtered_updates:
        filtered_updates = [clipped_updates[0]]
        filtered_trust = [trust_scores[0]]
    
    baseline_median = coordinate_median(filtered_updates)
    
    similarity_scores = []
    for update in filtered_updates:
        if np.linalg.norm(baseline_median) > 1e-8:
            cos_sim = np.dot(update, baseline_median) / (np.linalg.norm(update) * np.linalg.norm(baseline_median))
        else:
            cos_sim = 1.0
        similarity_scores.append(max(cos_sim, 0))
    
    similarity_scores = np.array(similarity_scores)
    similarity_mask = similarity_scores >= sim_threshold
    
    final_updates = [filtered_updates[i] for i in range(len(filtered_updates)) if similarity_mask[i]]
    final_trust = [filtered_trust[i] for i in range(len(filtered_trust)) if similarity_mask[i]]
    final_staleness = [staleness_weights[i] for i in range(len(staleness_weights)) if similarity_mask[i]]
    
    if not final_updates:
        final_updates = [filtered_updates[0]]
        final_trust = [filtered_trust[0]]
        final_staleness = [1.0]
    
    robust_delta = trimmed_mean(final_updates, trim_ratio)
    
    robust_delta = np.nan_to_num(robust_delta, nan=0.0, posinf=1e6, neginf=-1e6)
    
    trust_weights = compute_trust_weights(final_trust, trust_power)
    staleness_w = np.array(final_staleness)
    
    combined_weights = trust_weights * staleness_w
    combined_weights = np.nan_to_num(combined_weights, nan=0.0)
    combined_weights = combined_weights / (np.sum(combined_weights) + 1e-8)
    
    weighted_delta = np.zeros_like(robust_delta)
    for i, update in enumerate(final_updates):
        update_clean = np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)
        weighted_delta += combined_weights[i] * update_clean
    
    return weighted_delta
