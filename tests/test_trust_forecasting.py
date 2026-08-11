from pathlib import Path

from tfacd.analytics.trust_forecasting import forecast_series, forecast_trust_trend, predict_risk_trajectory

REAL_AUDIT_LOG = Path(__file__).resolve().parents[1] / "artifacts" / "trust_boundary" / "audit_log.jsonl"

RISING = [0.10, 0.22, 0.31, 0.44, 0.53, 0.61]
FALLING = [0.90, 0.78, 0.69, 0.55, 0.47, 0.38]


def test_rising_trend_forecast_slopes_upward():
    result = forecast_series(RISING, horizon=3)
    assert result.slope > 0
    assert result.n_observations == len(RISING)
    assert len(result.point_forecast) == 3
    # extrapolation should keep climbing past the last observed value
    assert result.point_forecast[0] > RISING[-1]
    assert result.point_forecast[-1] > result.point_forecast[0]


def test_falling_trend_forecast_slopes_downward():
    result = forecast_series(FALLING, horizon=3)
    assert result.slope < 0
    assert result.point_forecast[0] < FALLING[-1]
    assert result.point_forecast[-1] < result.point_forecast[0]


def test_uncertainty_band_widens_around_a_noisy_series():
    noisy = [0.5, 0.3, 0.7, 0.2, 0.6, 0.4]
    result = forecast_series(noisy, horizon=2)
    for point, lower, upper in zip(result.point_forecast, result.lower_band, result.upper_band):
        assert lower <= point <= upper
    assert result.upper_band[0] > result.lower_band[0]  # non-degenerate band


def test_single_observation_forecasts_flat_with_zero_band():
    result = forecast_series([0.42], horizon=3)
    assert result.slope == 0.0
    assert result.point_forecast == [0.42, 0.42, 0.42]
    assert result.lower_band == result.upper_band == result.point_forecast


def test_clip_keeps_forecast_within_score_bounds():
    steep_rise = [0.80, 0.90, 0.98, 1.0]
    result = forecast_series(steep_rise, horizon=3, clip=(0.0, 1.0))
    assert all(0.0 <= v <= 1.0 for v in result.point_forecast)
    assert all(0.0 <= v <= 1.0 for v in result.upper_band)


def test_forecast_trust_trend_on_real_audit_log_produces_one_forecast_per_agent():
    forecasts = forecast_trust_trend(REAL_AUDIT_LOG, horizon=2)
    assert forecasts  # the real fixture has several agents with scored entries
    for agent_id, result in forecasts.items():
        assert isinstance(agent_id, str)
        assert len(result.point_forecast) == 2
        assert result.n_observations > 0


def test_predict_risk_trajectory_on_real_audit_log_covers_same_agents_as_trust_trend():
    trust_forecasts = forecast_trust_trend(REAL_AUDIT_LOG)
    risk_forecasts = predict_risk_trajectory(REAL_AUDIT_LOG)
    # same shared extraction path (scores present/absent), so the agent sets match
    assert set(risk_forecasts) == set(trust_forecasts)
