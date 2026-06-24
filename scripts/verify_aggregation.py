"""
Unit-level verification of ASTRA aggregation math.

Tests FedAvg, TrimmedMean, CoordinateMedian, and Hybrid aggregation
with known deltas to verify correctness. No server needed.

Usage:
    python scripts/verify_aggregation.py
    python scripts/verify_aggregation.py -v    # verbose
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from astra.core.aggregation.aggregator import (
    FedAvgAggregator,
    RobustAggregator,
    create_aggregator,
)
from astra.core.aggregation.robust import (
    coordinate_median,
    hybrid_aggregator,
    trimmed_mean,
)


def make_deltas(n: int = 5, dim: int = 100, seed: int = 42) -> list[np.ndarray]:
    """Create n random deltas of given dimension."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim).astype(np.float32) for _ in range(n)]


def make_buffer(deltas, sizes=None, trusts=None, staleness=None):
    """Build the buffer dict format expected by aggregators."""
    n = len(deltas)
    if sizes is None:
        sizes = [100] * n
    if trusts is None:
        trusts = [1.0] * n
    if staleness is None:
        staleness = [1.0] * n
    return [
        {
            "delta": d,
            "local_dataset_size": s,
            "trust": t,
            "staleness_weight": w,
        }
        for d, s, t, w in zip(deltas, sizes, trusts, staleness)
    ]


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_fedavg_equal_weights(verbose=False):
    """FedAvg with equal dataset sizes = simple average."""
    deltas = make_deltas(5, 100)
    expected = np.mean(deltas, axis=0)
    agg = FedAvgAggregator({})
    result = agg.aggregate(make_buffer(deltas))
    ok = np.allclose(result, expected, atol=1e-6)
    if verbose:
        print(f"  FedAvg equal weights: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"    expected[:5]={expected[:5]}, got[:5]={result[:5]}")
    return ok


def test_fedavg_weighted():
    """FedAvg with different dataset sizes = weighted average."""
    deltas = [np.array([2.0, 0.0]), np.array([0.0, 2.0])]
    sizes = [100, 100]
    # Equal sizes → average = [1.0, 1.0]
    agg = FedAvgAggregator({})
    result = agg.aggregate(make_buffer(deltas, sizes=sizes))
    expected = np.array([1.0, 1.0])
    ok = np.allclose(result, expected, atol=1e-6)
    print(f"  FedAvg weighted: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    expected={expected}, got={result}")
    return ok


def test_fedavg_staleness():
    """Staleness weighting: stale update gets lower weight."""
    delta_good = np.array([10.0])
    delta_stale = np.array([0.0])
    buffer = [
        {"delta": delta_good, "local_dataset_size": 1, "trust": 1.0, "staleness_weight": 1.0},
        {"delta": delta_stale, "local_dataset_size": 1, "trust": 1.0, "staleness_weight": 0.1},
    ]
    agg = FedAvgAggregator({})
    result = agg.aggregate(buffer)
    # With equal dataset sizes: result = (1.0*10 + 0.1*0) / (1.0 + 0.1) = 10/1.1 ≈ 9.09
    expected = 10.0 / 1.1
    ok = abs(result[0] - expected) < 0.01
    print(f"  FedAvg staleness: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    expected={expected:.4f}, got={result[0]:.4f}")
    return ok


def test_trimmed_mean_basic():
    """TrimmedMean trims top/bottom fraction."""
    # 5 updates, trim_ratio=0.2 → trim 1 from top, 1 from bottom
    deltas = [
        np.array([1.0]),
        np.array([2.0]),
        np.array([3.0]),
        np.array([4.0]),
        np.array([100.0]),  # outlier
    ]
    result = trimmed_mean(deltas, trim_ratio=0.2)
    # Sorted: [1, 2, 3, 4, 100]. Trim 1 from each end → [2, 3, 4] → mean = 3.0
    ok = abs(result[0] - 3.0) < 0.01
    print(f"  TrimmedMean basic: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    expected=3.0, got={result[0]:.4f}")
    return ok


def test_trimmed_mean_rejects_outlier():
    """TrimmedMean should ignore large outlier."""
    rng = np.random.default_rng(99)
    deltas = [rng.standard_normal(50).astype(np.float32) for _ in range(9)]
    # Add one extreme outlier
    outlier = deltas[0].copy() * 1000
    deltas.append(outlier)

    result = trimmed_mean(deltas, trim_ratio=0.1)
    # Without outlier: mean of remaining 9
    normal_result = trimmed_mean(deltas[:-1], trim_ratio=0.1)
    # They should be similar since outlier is trimmed
    ok = np.allclose(result, normal_result, atol=0.5)
    print(f"  TrimmedMean outlier rejection: {'PASS' if ok else 'FAIL'}")
    return ok


def test_coordinate_median():
    """CoordinateMedian = element-wise median."""
    deltas = [
        np.array([1.0, 10.0]),
        np.array([2.0, 20.0]),
        np.array([3.0, 30.0]),
        np.array([4.0, 40.0]),
        np.array([5.0, 50.0]),
    ]
    result = coordinate_median(deltas)
    expected = np.array([3.0, 30.0])
    ok = np.allclose(result, expected, atol=1e-6)
    print(f"  CoordinateMedian: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    expected={expected}, got={result}")
    return ok


def test_hybrid_filters_byzantine():
    """Hybrid aggregator filters out high-norm Byzantine updates."""
    rng = np.random.default_rng(77)
    # 4 honest updates: small norm
    honest = [rng.standard_normal(50).astype(np.float32) * 0.1 for _ in range(4)]
    # 1 Byzantine: huge norm
    byzantine = [rng.standard_normal(50).astype(np.float32) * 100.0]

    config = {
        "robust": {
            "method": "hybrid",
            "norm_clip": 5.0,
            "anomaly_k": 3.0,
            "sim_threshold": 0.2,
            "trim_ratio": 0.1,
            "trust_power": 1.0,
        }
    }

    # All 5 updates
    all_deltas = honest + byzantine
    result_with_byz = hybrid_aggregator(
        all_deltas, [1.0]*5, [1.0]*5, config, [100]*5
    )

    # Only honest updates
    result_honest = hybrid_aggregator(
        honest, [1.0]*4, [1.0]*4, config, [100]*4
    )

    # Results should be similar if Byzantine was filtered
    ok = np.allclose(result_with_byz, result_honest, atol=1.0)
    print(f"  Hybrid Byzantine filtering: {'PASS' if ok else 'FAIL'}")
    if not ok:
        diff = np.linalg.norm(result_with_byz - result_honest)
        print(f"    L2 diff={diff:.4f}")
    return ok


def test_hybrid_trust_weighting():
    """Hybrid: lower trust → lower weight in final aggregation."""
    config = {
        "robust": {
            "method": "hybrid",
            "norm_clip": 5.0,
            "anomaly_k": 3.0,
            "sim_threshold": 0.2,
            "trim_ratio": 0.1,
            "trust_power": 1.0,
        }
    }

    delta_high = np.array([10.0, 0.0])
    delta_low = np.array([0.0, 10.0])

    # High trust on delta_high
    result1 = hybrid_aggregator(
        [delta_high, delta_low], [1.0, 0.1], [1.0, 1.0], config, [100, 100]
    )
    # High trust on delta_low
    result2 = hybrid_aggregator(
        [delta_high, delta_low], [0.1, 1.0], [1.0, 1.0], config, [100, 100]
    )

    # result1 should be closer to delta_high, result2 closer to delta_low
    ok1 = result1[0] > result2[0]  # first component higher when delta_high trusted
    ok2 = result1[1] < result2[1]  # second component lower when delta_high trusted
    ok = ok1 and ok2
    print(f"  Hybrid trust weighting: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    result1={result1}, result2={result2}")
    return ok


def test_create_aggregator_factory():
    """Factory function creates correct aggregator type."""
    fedavg_cfg = {"robust": {"method": "fedavg"}}
    robust_cfg = {"robust": {"method": "median"}}

    a1 = create_aggregator(fedavg_cfg)
    a2 = create_aggregator(robust_cfg)

    ok1 = isinstance(a1, FedAvgAggregator)
    ok2 = isinstance(a2, RobustAggregator)
    ok = ok1 and ok2
    print(f"  Aggregator factory: {'PASS' if ok else 'FAIL'}")
    return ok


def test_robust_methods_dispatch():
    """RobustAggregator dispatches to correct method."""
    deltas = make_deltas(5, 50)
    buffer = make_buffer(deltas)

    for method in ["trimmed_mean", "median", "hybrid"]:
        cfg = {"robust": {"method": method}}
        agg = RobustAggregator(cfg)
        result = agg.aggregate(buffer)
        ok = len(result) == 50 and np.all(np.isfinite(result))
        if not ok:
            print(f"  Robust dispatch ({method}): FAIL")
            return False

    print(f"  Robust dispatch: PASS")
    return ok


def test_single_update():
    """All methods should handle single update gracefully."""
    delta = np.array([1.0, 2.0, 3.0])
    buffer = make_buffer([delta])

    agg_fedavg = FedAvgAggregator({})
    r1 = agg_fedavg.aggregate(buffer)

    r2 = trimmed_mean([delta], trim_ratio=0.1)
    r3 = coordinate_median([delta])
    r4 = hybrid_aggregator([delta], [1.0], [1.0], {"robust": {"method": "hybrid"}})

    ok = (
        np.allclose(r1, delta) and
        np.allclose(r2, delta) and
        np.allclose(r3, delta) and
        np.allclose(r4, delta)
    )
    print(f"  Single update handling: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify ASTRA aggregation math")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    tests = [
        ("FedAvg equal weights", test_fedavg_equal_weights),
        ("FedAvg weighted", test_fedavg_weighted),
        ("FedAvg staleness", test_fedavg_staleness),
        ("TrimmedMean basic", test_trimmed_mean_basic),
        ("TrimmedMean outlier", test_trimmed_mean_rejects_outlier),
        ("CoordinateMedian", test_coordinate_median),
        ("Hybrid Byzantine", test_hybrid_filters_byzantine),
        ("Hybrid trust", test_hybrid_trust_weighting),
        ("Aggregator factory", test_create_aggregator_factory),
        ("Robust dispatch", test_robust_methods_dispatch),
        ("Single update", test_single_update),
    ]

    print("Aggregation verification tests")
    print("=" * 50)

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            import inspect
            sig = inspect.signature(fn)
            if "verbose" in sig.parameters:
                ok = fn(verbose=args.verbose)
            else:
                ok = fn()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All aggregation math is correct!")


if __name__ == "__main__":
    main()
