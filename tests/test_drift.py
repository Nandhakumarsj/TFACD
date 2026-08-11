from pathlib import Path

import numpy as np

from tfacd.analytics.drift import audit_log_drift, detect_drift_points, ftil_trust_drift

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_LOG = _REPO_ROOT / "artifacts" / "trust_boundary" / "audit_log.jsonl"
_FTIL_LOG = _REPO_ROOT / "artifacts" / "models" / "ftil_trust_log.jsonl"


def test_stable_series_does_not_fire():
    rng = np.random.default_rng(0)
    stable = list(np.clip(rng.normal(0.5, 0.05, 80), 0.0, 1.0))
    assert detect_drift_points(stable) == []


def test_step_change_fires_near_the_change_point():
    step = [0.2] * 30 + [0.8] * 30
    drift_points = detect_drift_points(step)
    assert len(drift_points) >= 1
    assert 28 <= drift_points[0] <= 35  # fires just after the true change at index 30


def test_audit_log_drift_runs_on_the_real_log():
    result = audit_log_drift(_AUDIT_LOG)
    assert "agent-well_behaved" in result
    assert "agent-risky" in result
    for per_field in result.values():
        assert set(per_field) == {"semantic_risk", "context_consistency", "behavioral_trust", "trust_value"}
        for drift_points in per_field.values():
            assert all(isinstance(i, int) for i in drift_points)


def test_ftil_trust_drift_runs_on_the_real_log():
    result = ftil_trust_drift(_FTIL_LOG)
    assert set(result) == {"0", "1", "2", "3", "4"}
    for drift_points in result.values():
        assert all(isinstance(i, int) for i in drift_points)
