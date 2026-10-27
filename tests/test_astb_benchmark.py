import json
from pathlib import Path

from tfacd.evaluation.astb_benchmark import (
    COMPONENT_ABLATIONS,
    DECISIONS,
    EvaluationPolicyAdapter,
    SecurityPolicyOracle,
    generate_scenarios,
    load_oracle_config,
    load_prediction_policy_config,
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
    oracle = SecurityPolicyOracle(load_oracle_config(root / "configs" / "astb_oracle.yaml"), policy, mapping)
    predictor = EvaluationPolicyAdapter(load_prediction_policy_config(root / "configs" / "astb_predictor.yaml"), policy)
    predictor.validate()
    return oracle, mapping, predictor


def test_generated_benchmark_is_deterministic_and_covers_taxonomy():
    oracle, _, _predictor = _oracle()
    a = generate_scenarios(200, seed=42, oracle=oracle)
    b = generate_scenarios(200, seed=42, oracle=oracle)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    categories = {x.attack_category for x in a}
    assert {"prompt_level", "tool", "context", "behavior", "execution", "control"} <= categories
    counts = {category: sum(x.attack_category == category for x in a) for category in categories}
    assert counts["control"] == 40
    assert all(counts[category] == 32 for category in ("prompt_level", "tool", "context", "behavior", "execution"))
    assert all(x.expected_decision in DECISIONS for x in a)


def test_ablation_reports_security_metrics_and_variant_names():
    oracle, mapping, predictor = _oracle()
    scenarios = generate_scenarios(150, seed=7, oracle=oracle)
    _, summary = run_ablation(scenarios, specs=COMPONENT_ABLATIONS, predictor_policy=predictor, threat_mapping=mapping)
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
    assert metrics["false_rejection_rate"] == 0.5
    assert metrics["uabr_numerator"] == 1
    assert metrics["uabr_denominator"] == 2
    assert metrics["far_numerator"] == 1
    assert metrics["far_denominator"] == 2
    assert metrics["frr_numerator"] == 1
    assert metrics["frr_denominator"] == 2
    assert metrics["sapr_numerator"] == 1
    assert metrics["sapr_denominator"] == 2
    assert metrics["safety_matrix_row_labels"] == ["unsafe", "legitimate"]
    assert metrics["safety_matrix_column_labels"] == ["ALLOW", "BLOCK", "HUMAN_REVIEW"]
    assert metrics["safety_confusion_matrix"] == [[1, 1, 0], [1, 1, 0]]


def test_three_class_metrics_use_unsafe_labels_not_oracle_block_subset():
    from tfacd.evaluation.astb_benchmark import _metrics

    rows = [
        # Unsafe case can be human-reviewed; it must stay in the unsafe denominator.
        {"expected_decision": "BLOCK", "predicted_decision": "BLOCK", "unsafe": True, "attack_category": "tool", "attack_type": "unauthorized_tool", "latency_ms": 1.0, "context_consistency_raw": 1.0},
        {"expected_decision": "BLOCK", "predicted_decision": "ALLOW", "unsafe": True, "attack_category": "tool", "attack_type": "unauthorized_tool", "latency_ms": 1.0, "context_consistency_raw": 1.0},
        {"expected_decision": "HUMAN_REVIEW", "predicted_decision": "HUMAN_REVIEW", "unsafe": True, "attack_category": "context", "attack_type": "target_substitution", "latency_ms": 1.0, "context_consistency_raw": 0.0},
        {"expected_decision": "ALLOW", "predicted_decision": "ALLOW", "unsafe": False, "attack_category": "control", "attack_type": "normal", "latency_ms": 1.0, "context_consistency_raw": 1.0},
        {"expected_decision": "HUMAN_REVIEW", "predicted_decision": "HUMAN_REVIEW", "unsafe": False, "attack_category": "control", "attack_type": "normal", "latency_ms": 1.0, "context_consistency_raw": 1.0},
    ]
    metrics = _metrics(rows)
    assert metrics["unsafe_count"] == 3
    assert metrics["legitimate_count"] == 2
    assert metrics["unsafe_action_blocking_rate"] == 1 / 3
    assert metrics["far_unsafe"] == 1 / 3
    assert metrics["unsafe_human_review_rate"] == 1 / 3
    assert metrics["safe_action_preservation_rate"] == 1.0
    assert metrics["non_unsafe_allow_rate"] == 1 / 2
    assert metrics["legitimate_human_review_rate"] == 1 / 2
    assert metrics["human_review_precision"] == 1.0
    assert metrics["human_review_recall"] == 1.0
    assert metrics["oracle_allow_preservation_rate"] == 1.0
    assert metrics["oracle_allow_human_review_rate"] == 0.0
    assert metrics["human_review_rate"] == 2 / 5
    assert metrics["decision_labels"] == ["ALLOW", "BLOCK", "HUMAN_REVIEW"]
    assert len(metrics["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in metrics["confusion_matrix"])
    assert metrics["by_attack_subtype"]["target_substitution"]["context_consistency_raw_mean"] == 0.0


def test_legitimate_control_context_scores_are_exact_and_oracle_has_review_cases():
    from tfacd.evaluation.astb_benchmark import _scenario_runtime_objects
    from tfacd.trust_boundary.context_consistency import score as context_score

    oracle, mapping, predictor = _oracle()
    scenarios = generate_scenarios(500, seed=42, oracle=oracle)
    controls = [scenario for scenario in scenarios if scenario.attack_category == "control"]
    assert controls
    assert any(s.expected_decision == "ALLOW" for s in controls)
    assert any(s.expected_decision == "HUMAN_REVIEW" for s in controls)

    raw_scores = {
        scenario.scenario_id: context_score(
            _scenario_runtime_objects(scenario, threat_mapping=mapping)[0],
            _scenario_runtime_objects(scenario, threat_mapping=mapping)[1],
        )
        for scenario in controls
    }
    assert set(raw_scores.values()) == {1.0}

    target_substitutions = [
        scenario for scenario in scenarios
        if scenario.attack_category == "context" and scenario.attack_type == "target_substitution"
    ]
    assert target_substitutions
    for scenario in target_substitutions:
        plan, context, _, _ = _scenario_runtime_objects(scenario, threat_mapping=mapping)
        assert context_score(plan, context) < oracle.context_consistency_threshold


def test_component_ablation_records_raw_context_diagnostics_and_human_review():
    oracle, mapping, predictor = _oracle()
    scenarios = generate_scenarios(200, seed=99, oracle=oracle)
    rows, summary = run_ablation(
        scenarios, specs=COMPONENT_ABLATIONS, predictor_policy=predictor, threat_mapping=mapping
    )
    full_rows = [row for row in rows if row["variant"] == "full_astb"]
    controls = [row for row in full_rows if row["attack_category"] == "control"]
    target_rows = [row for row in full_rows if row["attack_type"] == "target_substitution"]
    assert controls and target_rows
    assert all(row["context_consistency_raw"] == 1.0 for row in controls)
    assert all(row["context_consistency_raw"] < 1.0 for row in target_rows)
    assert any(row["predicted_decision"] == "ALLOW" for row in controls)
    assert all(row["decision_stage"] and row["decision_reason"] for row in controls)
    assert all(row["predicted_decision"] == "BLOCK" for row in target_rows)
    assert all(row["oracle_sha256"] is None for row in full_rows)  # no path hash supplied in this unit fixture
    assert all("software_versions" in row for row in full_rows)
    assert summary["variants"]["full_astb"]["by_attack_subtype"]["target_substitution"]["context_consistency_raw_min"] < 1.0
    assert summary["variant_components"]["full_astb"] == {
        "semantic": True, "context": True, "behavior": True, "history": True,
        "policy": True, "dynamic_trust": True,
    }


def test_prediction_path_is_separate_from_oracle_and_has_provenance():
    import inspect
    from tfacd.evaluation.astb_benchmark import evaluate_scenario, run_ablation

    assert "oracle" not in inspect.signature(evaluate_scenario).parameters
    assert "oracle" not in inspect.signature(run_ablation).parameters
    oracle, mapping, predictor = _oracle()
    scenarios = generate_scenarios(150, seed=314, oracle=oracle)
    rows, _ = run_ablation(
        scenarios, specs=COMPONENT_ABLATIONS,
        predictor_policy=predictor, threat_mapping=mapping,
    )
    required = {
        "decision_stage", "decision_reason", "stage_trace", "policy_status",
        "semantic_status", "context_status", "behavior_status", "history_status",
        "dynamic_trust_status", "human_review_status",
    }
    assert all(required <= set(row) for row in rows)


def test_reference_human_review_controls_are_sent_for_review_by_predictor_policy():
    from tfacd.evaluation.astb_benchmark import evaluate_scenario

    oracle, mapping, predictor = _oracle()
    scenario = next(
        s for s in generate_scenarios(300, seed=1207, oracle=oracle)
        if s.attack_category == "control" and s.expected_decision == "HUMAN_REVIEW"
    )

    class FixedSemanticRisk:
        def score(self, plan, context):
            return 0.10

    spec = next(s for s in COMPONENT_ABLATIONS if s.name == "full_astb")
    predicted, evidence, _ = evaluate_scenario(
        scenario, spec, predictor_policy=predictor,
        semantic_engine=FixedSemanticRisk(), trust_mapping=mapping,
    )
    assert predicted == "HUMAN_REVIEW"
    assert evidence["decision_stage"] == "human_review_policy"
    assert evidence["human_review_status"] == "required"


def test_review_metrics_use_reference_decision_not_unsafe_flag():
    from tfacd.evaluation.astb_benchmark import _metrics

    rows = [
        {"expected_decision": "ALLOW", "predicted_decision": "ALLOW", "unsafe": False, "attack_category": "control", "attack_type": "normal", "latency_ms": 1.0},
        {"expected_decision": "ALLOW", "predicted_decision": "HUMAN_REVIEW", "unsafe": False, "attack_category": "control", "attack_type": "normal", "latency_ms": 1.0},
        {"expected_decision": "HUMAN_REVIEW", "predicted_decision": "HUMAN_REVIEW", "unsafe": False, "attack_category": "control", "attack_type": "normal", "latency_ms": 1.0},
        {"expected_decision": "HUMAN_REVIEW", "predicted_decision": "ALLOW", "unsafe": False, "attack_category": "control", "attack_type": "normal", "latency_ms": 1.0},
        {"expected_decision": "BLOCK", "predicted_decision": "ALLOW", "unsafe": True, "attack_category": "tool", "attack_type": "unauthorized_tool", "latency_ms": 1.0},
        {"expected_decision": "BLOCK", "predicted_decision": "BLOCK", "unsafe": True, "attack_category": "tool", "attack_type": "unauthorized_tool", "latency_ms": 1.0},
    ]
    metrics = _metrics(rows)
    assert metrics["safe_action_preservation_rate"] == 0.5
    assert metrics["non_unsafe_allow_rate"] == 0.5
    assert metrics["human_review_recall"] == 0.5
    assert metrics["human_review_precision"] == 0.5
    assert metrics["oracle_non_allow_false_authorization_rate"] == 0.5
    assert metrics["far_unsafe"] == 0.5
    assert metrics["unsafe_action_blocking_rate"] == 0.5
    assert metrics["per_class"]["HUMAN_REVIEW"]["support"] == 2




def test_changing_only_oracle_label_cannot_change_prediction():
    from tfacd.evaluation.astb_benchmark import evaluate_scenario

    oracle, mapping, predictor = _oracle()
    scenario = next(
        s for s in generate_scenarios(200, seed=918, oracle=oracle)
        if s.attack_category == "control" and s.capability in predictor.known
    )
    alternate_label = "BLOCK" if scenario.expected_decision != "BLOCK" else "ALLOW"
    changed_label = scenario.__class__(**{
        **scenario.to_dict(), "expected_decision": alternate_label,
    })

    class FixedSemanticRisk:
        def score(self, plan, context):
            return 0.10

    spec = next(s for s in COMPONENT_ABLATIONS if s.name == "full_astb")
    original_prediction = evaluate_scenario(
        scenario, spec, predictor_policy=predictor,
        semantic_engine=FixedSemanticRisk(), trust_mapping=mapping,
    )
    changed_prediction = evaluate_scenario(
        changed_label, spec, predictor_policy=predictor,
        semantic_engine=FixedSemanticRisk(), trust_mapping=mapping,
    )
    assert original_prediction[0] == changed_prediction[0]
    assert original_prediction[1] == changed_prediction[1]
