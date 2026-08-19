"""Attack-scenario simulation: replays a normal-traffic baseline with a real
attack file injected mid-stream, paced by real PCAP capture timestamps
(streaming/live_source.py, streaming/scenario.py) instead of
run_streaming_demo.py's flat single-file held-out replay - see
configs/attack_scenarios.yaml for the built-in "network disconnect" /
"IP isolation" scenarios and docs on why PCAP timestamps (not the CSV's own
frame.time, which is empty on normal-traffic files) drive pacing.

This is simulation of the INPUT side only. Whether any capability actually
executes for real (vs SimulatedExecutor's "would execute" logging) is
entirely controlled by trust_boundary.executor.mode in --config - this script
never bypasses that switch. Reuses streaming/incident_runner.py's exact
per-incident pipeline wiring - the same one run_streaming_demo.py uses - so
there is exactly one implementation of "alert -> decision", not two that
could drift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tfacd.agentic.factory import build_decision_engine
from tfacd.agentic.history import EntityHistory
from tfacd.common.config import load_config
from tfacd.runtime.threat_context import ThreatContextGenerator
from tfacd.streaming.incident_runner import run_incident
from tfacd.streaming.pipeline import StreamingIDS
from tfacd.streaming.scenario import ScenarioRunner, load_scenarios
from tfacd.trust_boundary.audit import AuditLogger
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine
from tfacd.trust_boundary.boundary import AdaptiveSemanticTrustBoundary
from tfacd.trust_boundary.dynamic_trust import DynamicTrustScoreRegulator
from tfacd.trust_boundary.executor_factory import build_executor
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", required=True, help="scenario name from --scenarios-config")
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--scenarios-config", default="configs/attack_scenarios.yaml")
parser.add_argument("--speed", type=float, default=None, help="overrides every step's speed_multiplier in the scenario")
parser.add_argument("--max-incidents", type=int, default=10)
args = parser.parse_args()

config = load_config(args.config)
policy = load_config(config["runtime"]["trust_policy_path"])
tb_config = config["trust_boundary"]

scenarios = load_scenarios(args.scenarios_config)
if args.scenario not in scenarios:
    raise SystemExit(f"unknown scenario {args.scenario!r} - available: {sorted(scenarios)}")
scenario = scenarios[args.scenario]
if args.speed is not None:
    for step in scenario.steps:
        step.speed_multiplier = args.speed

print(f"Scenario: {scenario.name}\n  {scenario.description.strip()}")
if scenario.expected_capability is not None:
    print(f"  expected_capability (informational only): {scenario.expected_capability}")
else:
    print("  diagnostic scenario - no expected_capability to check for")
for step in scenario.steps:
    print(f"  step: {step.label:<28} max_records={step.max_records} speed_multiplier={step.speed_multiplier}x")

print("\n*** This is simulated INPUT pacing, not simulated response - whether any capability actually executes for "
      f"real is controlled entirely by trust_boundary.executor.mode ({tb_config.get('executor', {}).get('mode', 'simulate')!r} "
      "in this config), not by this script. ***\n")

print("Verifying and loading the certified model...")
ids = StreamingIDS.from_config(config)
print(f"Loaded {config['streaming']['model_path']} ({len(ids.classes)} classes)\n")

runner = ScenarioRunner(scenario)
alerts, stats = ids.run(runner)
print(f"Replayed the scenario: records_read={stats.records_read} alerts_emitted={stats.alerts_emitted} "
      f"alerts_suppressed={stats.alerts_suppressed} elapsed={stats.total_seconds:.1f}s (includes real pacing sleeps)\n")

incidents = alerts[: args.max_incidents]
if len(alerts) > args.max_incidents:
    print(f"Feeding the first {args.max_incidents} of {len(alerts)} alerts through the full agentic + trust-boundary path.\n")

output_dir = Path("artifacts/scenarios") / scenario.name
history = EntityHistory(persist_path=output_dir / "history.jsonl")
threat_context_generator = ThreatContextGenerator(config["runtime"]["threat_context_mapping"], known_classes=ids.classes)
decision_engine = build_decision_engine(config, history)
executor = build_executor(config)
if executor.mode == "production":
    print("*** trust_boundary.executor.mode=\"production\": REAL response actions will be taken for this scenario. ***\n")
boundary = AdaptiveSemanticTrustBoundary(
    history=history,
    policy=policy,
    preprocessing_config=tb_config,
    trust_regulator=DynamicTrustScoreRegulator(
        tb_config["weight_semantic_risk"], tb_config["weight_context_consistency"], tb_config["weight_behavioral_trust"], tb_config["trust_level_thresholds"]
    ),
    semantic_risk_engine=SemanticRiskEngine(model_name=tb_config["sbert_model_name"]),
    behavioral_trust_engine=BehavioralTrustEngine(high_risk_capabilities=set(policy["capability_whitelist"]["high_risk"]), ema_alpha=tb_config["ema_alpha"]),
    audit_logger=AuditLogger(output_dir / "audit_log.jsonl"),
    executor=executor,
)

alert_type_counts: dict[str, int] = {}
expected_capability_seen = False
for alert in incidents:
    context, decision = run_incident(
        alert, threat_context_generator=threat_context_generator, decision_engine=decision_engine, boundary=boundary, agent_id=f"scenario_{scenario.name}",
    )
    alert_type_counts[alert.attack_type] = alert_type_counts.get(alert.attack_type, 0) + 1
    if scenario.expected_capability is not None and scenario.expected_capability in decision.executed_actions:
        expected_capability_seen = True
    print(
        f"{alert.attack_type:<22} confidence={alert.confidence:.3f} severity={context.severity:<13} "
        f"trust_level={decision.trust_level} executor_mode={decision.executor_mode} executed={decision.executed_actions}"
    )

print(f"\nalert_type counts among the {len(incidents)} incident(s) fed through the pipeline: {alert_type_counts}")
if scenario.expected_capability is not None:
    print(f"expected_capability {scenario.expected_capability!r} was {'proposed and executed' if expected_capability_seen else 'NOT executed'} "
          "at some point in this run (reported, not forced - the trust boundary decided this for real on the replayed evidence).")
print(f"\naudit log: {output_dir / 'audit_log.jsonl'}")
print(f"history: {output_dir / 'history.jsonl'}")
