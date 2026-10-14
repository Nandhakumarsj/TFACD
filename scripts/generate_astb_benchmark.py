from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tfacd.evaluation.astb_benchmark import (
    SecurityPolicyOracle,
    generate_scenarios,
    load_oracle_config,
    load_threat_mapping,
)
from tfacd.common.config import load_config

parser = argparse.ArgumentParser(description="Generate the deterministic ASTB-IIoT Security Evaluation Dataset.")
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--oracle", default="configs/astb_oracle.yaml")
parser.add_argument("--output-dir", default="artifacts/astb_benchmark")
parser.add_argument("--count", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

config = load_config(args.config)
policy = load_config(config["runtime"]["trust_policy_path"])
threat_mapping = load_threat_mapping(config["runtime"]["threat_context_mapping"])
oracle = SecurityPolicyOracle(load_oracle_config(args.oracle), policy, threat_mapping)
oracle.validate()
scenarios = generate_scenarios(args.count, seed=args.seed, oracle=oracle)

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
jsonl = out / "astb_iiot_security_evaluation.jsonl"
csv_path = out / "astb_iiot_security_evaluation.csv"
with jsonl.open("w", encoding="utf-8") as handle:
    for row in scenarios:
        handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")

rows = [row.to_dict() for row in scenarios]
fields = list(rows[0])
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        flattened = dict(row)
        flattened["parameters"] = json.dumps(row["parameters"], sort_keys=True)
        writer.writerow(flattened)

manifest = {
    "oracle_version": load_oracle_config(args.oracle)["oracle_version"],
    "oracle_path": str(Path(args.oracle)),
    "seed": args.seed,
    "count": len(scenarios),
    "taxonomy": {k: len(v) for k, v in __import__("tfacd.evaluation.astb_benchmark", fromlist=["ATTACK_TAXONOMY"]).ATTACK_TAXONOMY.items()},
    "label_distribution": {},
}
for row in scenarios:
    manifest["label_distribution"][row.expected_decision] = manifest["label_distribution"].get(row.expected_decision, 0) + 1
(out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

print(f"Generated {len(scenarios)} scenarios")
print(f"JSONL : {jsonl.resolve()}")
print(f"CSV   : {csv_path.resolve()}")
print("Labels:", manifest["label_distribution"])
