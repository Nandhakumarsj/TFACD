from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tfacd.common.config import load_config
from tfacd.evaluation.astb_benchmark import (
    ABLATIONS,
    COMPONENT_ABLATIONS,
    BenchmarkScenario,
    SecurityPolicyOracle,
    load_oracle_config,
    load_threat_mapping,
    run_ablation,
)

parser = argparse.ArgumentParser(description="Run A-D ASTB ablation and component-wise ASTB ablation.")
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--oracle", default="configs/astb_oracle.yaml")
parser.add_argument("--dataset", default="artifacts/astb_benchmark/astb_iiot_security_evaluation.jsonl")
parser.add_argument("--output-dir", default="artifacts/astb_benchmark")
args = parser.parse_args()

config = load_config(args.config)
policy = load_config(config["runtime"]["trust_policy_path"])
threat_mapping = load_threat_mapping(config["runtime"]["threat_context_mapping"])
oracle = SecurityPolicyOracle(load_oracle_config(args.oracle), policy, threat_mapping)

scenarios = []
for line in Path(args.dataset).read_text(encoding="utf-8").splitlines():
    if line.strip():
        scenarios.append(BenchmarkScenario(**json.loads(line)))

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
for label, specs in (("coarse", ABLATIONS), ("component", COMPONENT_ABLATIONS)):
    rows, summary = run_ablation(scenarios, specs=specs, oracle=oracle, threat_mapping=threat_mapping)
    (out / f"{label}_ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (out / f"{label}_ablation_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== {label.upper()} ABLATION ===")
    print(f"{'variant':<24} {'F1':>8} {'FAR':>8} {'UABR':>8} {'SAPR':>8} {'PI success':>12} {'Tool allow':>11} {'p95 ms':>10}")
    for name, metric in summary["variants"].items():
        print(
            f"{name:<24} {metric['macro_f1']:>8.3f} {metric['false_allow_rate']:>8.3f} "
            f"{metric['unsafe_action_blocking_rate']:>8.3f} {metric['safe_action_preservation_rate']:>8.3f} "
            f"{metric['prompt_injection_success_rate']:>12.3f} {metric['unauthorized_tool_call_rate']:>11.3f} "
            f"{metric['latency_p95_ms']:>10.2f}"
        )
