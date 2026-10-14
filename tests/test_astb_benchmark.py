import json
from pathlib import Path

from tfacd.evaluation.astb_benchmark import (
    COMPONENT_ABLATIONS,
    DECISIONS,
    SecurityPolicyOracle,
    generate_scenarios,
    load_oracle_config,
    load_threat_mapping,
    run_ablation,
)


def _oracle():
    root = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(root / "src"))
    from tfacd.common.config import load_config
    config = load_config(root / "configs" / "edge_iiot.yaml")
    policy = load_config(root / config["runtime"]["trust_policy_path"])
    mapping = load_threat_mapping(root / config["runtime"]["threat_context_mapping"])
    return SecurityPolicyOracle(load_oracle_config(root / "configs" / "astb_oracle.yaml"), policy, mapping), mapping


def test_generated_benchmark_is_deterministic_and_covers_taxonomy():
    oracle, _ = _oracle()
    a = generate_scenarios(200, seed=42, oracle=oracle)
    b = generate_scenarios(200, seed=42, oracle=oracle)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    categories = {x.attack_category for x in a}
    assert {"prompt_level", "tool", "context", "behavior", "execution", "control"} <= categories
    assert all(x.expected_decision in DECISIONS for x in a)


def test_ablation_reports_security_metrics_and_variant_names():
    oracle, mapping = _oracle()
    scenarios = generate_scenarios(150, seed=7, oracle=oracle)
    _, summary = run_ablation(scenarios, specs=COMPONENT_ABLATIONS, oracle=oracle, threat_mapping=mapping)
    assert list(summary["variants"]) == [x.name for x in COMPONENT_ABLATIONS]
    for metrics in summary["variants"].values():
        assert 0.0 <= metrics["false_allow_rate"] <= 1.0
        assert 0.0 <= metrics["unsafe_action_blocking_rate"] <= 1.0
        assert 0.0 <= metrics["safe_action_preservation_rate"] <= 1.0
        assert metrics["latency_p95_ms"] >= 0.0


def test_false_allow_rate_uses_unsafe_count_as_denominator():
    from tfacd.evaluation.astb_benchmark import _metrics
    rows = [
        {"expected_decision": "BLOCK", "predicted_decision": "ALLOW", "unsafe": True, "attack_category": "tool", "latency_ms": 1.0},
        {"expected_decision": "BLOCK", "predicted_decision": "BLOCK", "unsafe": True, "attack_category": "tool", "latency_ms": 1.0},
        {"expected_decision": "ALLOW", "predicted_decision": "ALLOW", "unsafe": False, "attack_category": "control", "latency_ms": 1.0},
        {"expected_decision": "ALLOW", "predicted_decision": "BLOCK", "unsafe": False, "attack_category": "control", "latency_ms": 1.0},
    ]
    metrics = _metrics(rows)
    assert metrics["false_allow_rate"] == 0.5
    assert metrics["unsafe_action_blocking_rate"] == 0.5
    assert metrics["safe_action_preservation_rate"] == 0.5
