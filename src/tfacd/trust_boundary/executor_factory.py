"""Config-driven selection between SimulatedExecutor (default, today's only
behavior) and ProductionExecutor (real OS-level actions) - the trust_boundary
analogue of agentic/factory.py::build_decision_engine. Every script that
constructs an AdaptiveSemanticTrustBoundary should build its executor through
this function rather than constructing SimulatedExecutor/ProductionExecutor
directly, so "simulate" stays the one actual default across every entry point.
"""

from __future__ import annotations

from typing import Any

from tfacd.trust_boundary.capability_enforcement import CapabilityExecutor, SimulatedExecutor
from tfacd.trust_boundary.production_executor import DEFAULT_PROTECTED_TARGETS, ProductionExecutor


def build_executor(config: dict[str, Any]) -> CapabilityExecutor:
    executor_cfg = config.get("trust_boundary", {}).get("executor", {})
    mode = executor_cfg.get("mode", "simulate")

    if mode == "simulate":
        return SimulatedExecutor()

    if mode == "production":
        return ProductionExecutor(
            protected_targets=executor_cfg.get("protected_targets", list(DEFAULT_PROTECTED_TARGETS)),
            action_log_path=executor_cfg.get("action_log_path", "artifacts/trust_boundary/production_action_log.jsonl"),
            session_rotation_log_path=executor_cfg.get("session_rotation_log_path", "artifacts/trust_boundary/session_rotation_requests.jsonl"),
        )

    raise ValueError(f"unknown trust_boundary.executor.mode={mode!r} (expected 'simulate' or 'production')")
