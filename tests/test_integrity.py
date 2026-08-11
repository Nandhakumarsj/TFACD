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
