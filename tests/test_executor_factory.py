import pytest

from tfacd.trust_boundary.capability_enforcement import SimulatedExecutor
from tfacd.trust_boundary.executor_factory import build_executor
from tfacd.trust_boundary.production_executor import ProductionExecutor


def test_absent_executor_section_defaults_to_simulated():
    assert isinstance(build_executor({}), SimulatedExecutor)


def test_explicit_simulate_mode():
    config = {"trust_boundary": {"executor": {"mode": "simulate"}}}
    assert isinstance(build_executor(config), SimulatedExecutor)


def test_production_mode_builds_production_executor():
    config = {"trust_boundary": {"executor": {"mode": "production"}}}
    executor = build_executor(config)
    assert isinstance(executor, ProductionExecutor)


def test_production_mode_passes_through_protected_targets():
    config = {"trust_boundary": {"executor": {"mode": "production", "protected_targets": ["10.0.0.1/32"]}}}
    executor = build_executor(config)
    assert executor.protected_targets == ["10.0.0.1/32"]


def test_unknown_mode_raises():
    config = {"trust_boundary": {"executor": {"mode": "not-a-real-mode"}}}
    with pytest.raises(ValueError, match="unknown"):
        build_executor(config)
