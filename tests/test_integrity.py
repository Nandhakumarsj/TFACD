import numpy as np

from tfacd.integrity.aggregation import coordinate_median, trimmed_mean, weighted_average
from tfacd.integrity.detector import PCAClusterEMAFilter
from tfacd.integrity.update_validation import validate_update


def state(value: float):
    return {"w": np.full((2, 2), value, dtype=np.float32)}


def test_validation_rejects_nan():
    candidate = state(1.0)
    candidate["w"][0, 0] = np.nan
    result = validate_update(candidate, state(0.0), max_abs_parameter=100, max_update_norm_ratio=100)
    assert not result.accepted


def test_aggregators_reduce_outlier_influence():
    states = [state(1.0), state(1.1), state(100.0)]
    mean = weighted_average(states, [1, 1, 1])["w"][0, 0]
    median = coordinate_median(states)["w"][0, 0]
    trimmed = trimmed_mean(states, 0.34)["w"][0, 0]
    assert mean > 30
    assert 1.0 <= median <= 1.1
    assert 1.0 <= trimmed <= 1.1


def test_detector_flags_far_update():
    rng = np.random.default_rng(0)
    benign = rng.normal(0, 0.05, size=(5, 20))
    malicious = np.full((1, 20), 10.0)
    vectors = np.vstack([benign, malicious])
    detector = PCAClusterEMAFilter(ema_alpha=1.0, reject_below_trust=0.5)
    result = detector.detect([str(i) for i in range(6)], vectors)
    assert result.benign_mask.sum() >= 3
    assert not result.benign_mask[-1]


def test_detector_ood_score_ranks_the_outlier_highest():
    """ood_scores is the continuous, memoryless signal - it should rank the
    known-malicious client highest regardless of what round_scores/trust_scores say."""
    rng = np.random.default_rng(0)
    benign = rng.normal(0, 0.05, size=(5, 20))
    malicious = np.full((1, 20), 10.0)
    vectors = np.vstack([benign, malicious])
    detector = PCAClusterEMAFilter(ema_alpha=1.0, reject_below_trust=0.5)
    result = detector.detect([str(i) for i in range(6)], vectors)

    assert result.ood_scores[-1] == max(result.ood_scores)
    assert result.ood_scores[-1] > 1.0  # farther than the cohort's median distance


def test_detector_metrics_report_cluster_method_and_quality():
    rng = np.random.default_rng(0)
    vectors = rng.normal(0, 1.0, size=(6, 10))
    detector = PCAClusterEMAFilter(cluster_method="agglomerative")
    result = detector.detect([str(i) for i in range(6)], vectors)

    assert result.metrics.cluster_method == "agglomerative"
    assert result.metrics.n_clients == 6
    assert 0.0 <= result.metrics.explained_variance_ratio <= 1.0
    assert not result.metrics.degenerate


def test_detector_degenerate_cohort_reports_zero_ood_without_crashing():
    """All-identical updates would otherwise divide distances by a ~zero median."""
    vectors = np.ones((5, 10), dtype=np.float64)
    detector = PCAClusterEMAFilter()
    result = detector.detect([str(i) for i in range(5)], vectors)

    assert np.all(result.ood_scores == 0.0)
    assert result.metrics.degenerate


def test_detector_too_few_clients_still_advances_trust_history():
    detector = PCAClusterEMAFilter(ema_alpha=0.5)
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = detector.detect(["a", "b"], vectors)

    assert result.metrics.cluster_method == "none:too-few-clients"
    assert result.metrics.degenerate
    assert np.all(result.benign_mask)
    assert detector.history == {"a": 1.0, "b": 1.0}
