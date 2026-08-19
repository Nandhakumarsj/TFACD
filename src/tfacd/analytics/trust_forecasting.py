"""Forecasts near-term trust/risk trajectories from the trust-boundary audit
log. Per-agent series here are short (a handful to a few dozen points) and
have no seasonality, so an OLS trend line with a residual-std band is used
instead of a heavier statsmodels/prophet model - that would be overkill for
data this thin and would invite false confidence in the forecast.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from tfacd.runtime.contracts import AuditEntry, TrustScores

_DEFAULT_AUDIT_LOG = "artifacts/trust_boundary/audit_log.jsonl"
_DEFAULT_HORIZON = 3


@dataclass
class ForecastResult:
    point_forecast: list[float]  # next `horizon` values, in order
    lower_band: list[float]
    upper_band: list[float]
    slope: float  # per-step trend used to extrapolate: >0 rising, <0 falling
    n_observations: int


def forecast_series(values: Sequence[float], horizon: int = _DEFAULT_HORIZON, band_z: float = 1.0, clip: tuple[float, float] | None = None) -> ForecastResult:
    """OLS trend line over `values` (index order = time), extrapolated `horizon`
    steps ahead with a constant +/- band_z*residual_std uncertainty band.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    array = np.asarray(values, dtype=np.float64)
    n = array.size
    if n == 0:
        raise ValueError("values must be non-empty")
    if n == 1:
        # One point has no trend to fit - flat-line forecast, zero-width band
        # rather than a fabricated uncertainty estimate.
        flat = float(array[0])
        return ForecastResult([flat] * horizon, [flat] * horizon, [flat] * horizon, 0.0, n)

    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, array, 1)
    residuals = array - (slope * t + intercept)
    # With exactly 2 points the line fits exactly (zero residuals) - an honest
    # zero-width band, not a bug: there is no residual signal yet to measure.
    residual_std = float(np.std(residuals))

    future_t = np.arange(n, n + horizon, dtype=np.float64)
    point = slope * future_t + intercept
    lower = point - band_z * residual_std
    upper = point + band_z * residual_std

    if clip is not None:
        lo, hi = clip
        point = np.clip(point, lo, hi)
        lower = np.clip(lower, lo, hi)
        upper = np.clip(upper, lo, hi)

    return ForecastResult(point.tolist(), lower.tolist(), upper.tolist(), float(slope), n)


def _score_series_by_agent(path: str | Path, extractor: Callable[[TrustScores], float]) -> dict[str, list[float]]:
    """Groups one named TrustScores field by agent_id, in timestamp order.
    Entries with scores=None (rejected at preprocessing/deterministic_controls,
    before any trust score was ever computed) are skipped, not fabricated as
    zero - same distinction boundary.py itself makes.
    """
    by_agent: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = AuditEntry.model_validate_json(line)
        if entry.decision.scores is None:
            continue
        agent_id = entry.agent_id or "unknown"
        by_agent[agent_id].append((entry.timestamp, extractor(entry.decision.scores)))
    return {agent_id: [value for _, value in sorted(points, key=lambda p: p[0])] for agent_id, points in by_agent.items()}


def forecast_trust_trend(path: str | Path = _DEFAULT_AUDIT_LOG, horizon: int = _DEFAULT_HORIZON, band_z: float = 1.0) -> dict[str, ForecastResult]:
    """Trust Trend Forecasting: near-term trust_value (T) trajectory per agent_id.
    Agents with no scored entries in the log are absent from the result, not
    zero-filled.
    """
    series = _score_series_by_agent(path, lambda scores: scores.trust_value)
    return {agent_id: forecast_series(values, horizon=horizon, band_z=band_z, clip=(0.0, 1.0)) for agent_id, values in series.items()}


def predict_risk_trajectory(path: str | Path = _DEFAULT_AUDIT_LOG, horizon: int = _DEFAULT_HORIZON, band_z: float = 1.0) -> dict[str, ForecastResult]:
    """Risk Trajectory Prediction: near-term semantic_risk (Rs) trajectory per
    agent_id. Same shared core as forecast_trust_trend, different score field.
    """
    series = _score_series_by_agent(path, lambda scores: scores.semantic_risk)
    return {agent_id: forecast_series(values, horizon=horizon, band_z=band_z, clip=(0.0, 1.0)) for agent_id, values in series.items()}
