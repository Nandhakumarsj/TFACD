"""End-to-end streaming demo: closes the loop across both architecture
diagrams. Replays held-out rows through the certified IDS, feeds resulting
alerts through the Threat Context Generator -> Agentic Decision Engine ->
Adaptive Semantic Trust Boundary, and logs to a dedicated audit trail
scripts/generate_security_dashboard.py / analytics/kpi.py can read directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from tfacd.agentic.factory import build_decision_engine
from tfacd.agentic.history import EntityHistory
from tfacd.common.config import load_config
from tfacd.data.preprocess import heldout_indices
from tfacd.runtime.threat_context import ThreatContextGenerator
from tfacd.streaming.incident_runner import run_incident
from tfacd.streaming.pipeline import StreamingIDS
from tfacd.streaming.sources import CsvReplaySource, InMemorySource
from tfacd.trust_boundary.audit import AuditLogger
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine
from tfacd.trust_boundary.boundary import AdaptiveSemanticTrustBoundary
from tfacd.trust_boundary.dynamic_trust import DynamicTrustScoreRegulator
from tfacd.trust_boundary.executor_factory import build_executor
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument(
    "--all-rows", action="store_true",
    help="Replay arbitrary rows (may include rows the model trained on) instead of the held-out test split. "
    "Any accuracy shown is then NOT a generalization metric.",
)
args = parser.parse_args()

config = load_config(args.config)
streaming_cfg = config["streaming"]
policy = load_config(config["runtime"]["trust_policy_path"])
tb_config = config["trust_boundary"]

print("Verifying and loading the certified model...")
ids = StreamingIDS.from_config(config)
print(f"Loaded {streaming_cfg['model_path']} ({len(ids.classes)} classes)")

max_records = streaming_cfg.get("max_records")
if streaming_cfg.get("holdout_only", True) and not args.all_rows:
    idx_test = heldout_indices(config)
    rng = np.random.default_rng(streaming_cfg["replay_seed"])
    sample = np.sort(rng.choice(idx_test, size=min(max_records, len(idx_test)), replace=False)) if max_records else idx_test
    row_indices = sample.tolist()
    print(f"Replaying {len(row_indices)} held-out rows already scored by Gate 2/3 - a deviation from the offline "
          "macro-F1 below would indicate a pipeline bug, not generalization. Full-file scan to locate them (one-time cost)...")
else:
    print("*** --all-rows: replaying ARBITRARY rows, which may include rows the model trained on. Any accuracy "
          "shown is NOT a generalization metric - see scripts/evaluate_checkpoint.py for the authoritative held-out number. ***")
    row_indices = None

scan_source = CsvReplaySource(config["data"]["raw_csv"], chunk_size=streaming_cfg["batch_size"], max_records=max_records if row_indices is None else None, row_indices=row_indices)
records = list(scan_source.records())  # materialize once - reused below for both the accuracy pass and the real demo pass
print(f"Scanned {len(records)} rows.\n")

# --- Accuracy pass: emit_normal=True/min_confidence=0.0 regardless of the real
# config, so per-class support/macro-F1 reflect every prediction, not just the
# alerts the configured filtering would forward downstream. ---
eval_ids = StreamingIDS(ids.extractor, ids.model, ids.classes, device=ids.device, emit_normal=True, min_confidence=0.0)
eval_alerts, eval_stats = eval_ids.run(InMemorySource(records))
print(f"IDS inference: records_read={eval_stats.records_read} read={eval_stats.read_seconds:.2f}s "
      f"transform={eval_stats.transform_seconds:.2f}s inference={eval_stats.inference_seconds:.2f}s "
      f"-> {eval_stats.records_per_second:.1f} records/sec")

if row_indices is not None:
    output_dir = Path(config["data"]["output_dir"])
    idx_test_list = idx_test.tolist()
    position_by_index = {value: position for position, value in enumerate(idx_test_list)}
    true_labels = np.load(output_dir / "prepared.npz")["y_test"][[position_by_index[i] for i in row_indices]]
    true_names = [ids.classes[i] for i in true_labels]
    predicted_names = [a.attack_type for a in eval_alerts]
    report = classification_report(true_names, predicted_names, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    print(f"\nsubsample macro_f1={macro_f1:.4f} (n={len(true_names)} - noisy for rare classes at this sample size; "
          "scripts/evaluate_checkpoint.py on the full test set is the authoritative number)")
    for class_name, metrics in sorted(report.items(), key=lambda kv: kv[1]["support"] if isinstance(kv[1], dict) else 0, reverse=True):
        if isinstance(metrics, dict) and "support" in metrics and class_name not in ("macro avg", "weighted avg"):
            print(f"  {class_name:<22} support={int(metrics['support']):>4} recall={metrics['recall']:.3f}")

# --- Real demo pass: respects the configured emit_normal/min_confidence, and
# is what feeds the downstream Threat Context Generator -> Agentic Decision
# Engine -> Adaptive Semantic Trust Boundary. ---
real_ids = StreamingIDS(ids.extractor, ids.model, ids.classes, device=ids.device, emit_normal=streaming_cfg["emit_normal"], min_confidence=streaming_cfg["min_confidence"])
real_alerts, _ = real_ids.run(InMemorySource(records))
print(f"\n{len(real_alerts)} alert(s) at the configured emit_normal/min_confidence settings.")

max_incidents = streaming_cfg.get("max_incidents", 10)
incidents = real_alerts[:max_incidents]
if len(real_alerts) > max_incidents:
    print(f"Feeding the first {max_incidents} of {len(real_alerts)} alerts through the full agentic + trust-boundary "
          f"path (entity_action_quota_per_hour={tb_config['entity_action_quota_per_hour']} is a real security control, "
          "not raised to let a longer run through - a longer run would demonstrate it firing on purpose).")

history = EntityHistory(persist_path=Path("artifacts/streaming/history.jsonl"))
threat_context_generator = ThreatContextGenerator(
    config["runtime"]["threat_context_mapping"], known_classes=ids.classes,
)
decision_engine = build_decision_engine(config, history)
executor = build_executor(config)
if executor.mode == "production":
    print("*** trust_boundary.executor.mode=\"production\": REAL response actions will be taken (see "
          "trust_boundary/production_executor.py) - not simulated. ***")
boundary = AdaptiveSemanticTrustBoundary(
    history=history,
    policy=policy,
    preprocessing_config=tb_config,
    trust_regulator=DynamicTrustScoreRegulator(
        tb_config["weight_semantic_risk"], tb_config["weight_context_consistency"], tb_config["weight_behavioral_trust"], tb_config["trust_level_thresholds"]
    ),
    semantic_risk_engine=SemanticRiskEngine(model_name=tb_config["sbert_model_name"]),
    behavioral_trust_engine=BehavioralTrustEngine(high_risk_capabilities=set(policy["capability_whitelist"]["high_risk"]), ema_alpha=tb_config["ema_alpha"]),
    audit_logger=AuditLogger(Path("artifacts/streaming/audit_log.jsonl")),
    executor=executor,
)

print()
for alert in incidents:
    context, decision = run_incident(
        alert, threat_context_generator=threat_context_generator, decision_engine=decision_engine, boundary=boundary, agent_id="streaming_ids_v1",
    )
    print(
        f"{alert.attack_type:<22} confidence={alert.confidence:.3f} severity={context.severity:<13} "
        f"trust_level={decision.trust_level} executed={decision.executed_actions}"
    )

print("\naudit log: artifacts/streaming/audit_log.jsonl")
print("history: artifacts/streaming/history.jsonl")
