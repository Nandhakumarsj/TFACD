"""CLI entry point for Trust Trend Forecasting / Risk Trajectory Prediction
(analytics/trust_forecasting.py) - previously tested but never invoked outside
tests/test_trust_forecasting.py, per an architecture audit. Reads a real trust-
boundary audit log and prints each agent's near-term forecast.
"""

from __future__ import annotations

import argparse

from tfacd.analytics.trust_forecasting import forecast_trust_trend, predict_risk_trajectory

parser = argparse.ArgumentParser()
parser.add_argument("--audit-log", default="artifacts/trust_boundary/audit_log.jsonl")
parser.add_argument("--horizon", type=int, default=3, help="number of future steps to forecast")
args = parser.parse_args()


def _print_section(title: str, results: dict) -> None:
    print(f"\n=== {title} ===")
    if not results:
        print("  (no agent has scored entries in this log yet)")
        return
    for agent_id, forecast in sorted(results.items()):
        point = ", ".join(f"{v:.3f}" for v in forecast.point_forecast)
        band = ", ".join(f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(forecast.lower_band, forecast.upper_band))
        trend = "rising" if forecast.slope > 1e-6 else "falling" if forecast.slope < -1e-6 else "flat"
        print(f"  {agent_id:<22} n={forecast.n_observations:<3} slope={forecast.slope:+.4f} ({trend})")
        print(f"  {'':<22} next {len(forecast.point_forecast)}: {point}")
        print(f"  {'':<22} band: {band}")


trust = forecast_trust_trend(args.audit_log, horizon=args.horizon)
risk = predict_risk_trajectory(args.audit_log, horizon=args.horizon)

_print_section("Trust Trend Forecasting (T)", trust)
_print_section("Risk Trajectory Prediction (Rs)", risk)

print(f"\n{len(trust)} agent(s) with a scored history in {args.audit_log}")
print("Caveat: an OLS trend line over a short, noisy series - not a claim of future certainty. See forecast_series()'s docstring.")
