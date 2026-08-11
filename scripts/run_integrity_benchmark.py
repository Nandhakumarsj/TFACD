from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tfacd.common.config import load_config
from tfacd.common.reproducibility import seed_everything
from tfacd.integrity.benchmark import run_benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--malicious-client", type=int, action="append", default=None)
parser.add_argument("--max-samples-per-client", type=int, default=15000)
parser.add_argument("--init-checkpoint", default=None, help="Warm-start the global model (e.g. artifacts/models/centralized_best.pt)")
parser.add_argument("--output", default="artifacts/integrity/benchmark_report.json")
args = parser.parse_args()

config = load_config(args.config)
seed_everything(int(config.get("seed", 42)))
malicious = tuple(args.malicious_client) if args.malicious_client else (0,)


def report_progress(scenario: str, mode: str, round_index: int, num_rounds: int) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {scenario:<15} {mode:<18} round {round_index + 1}/{num_rounds} done")


report = run_benchmark(
    config,
    num_rounds=args.rounds,
    malicious_client_ids=malicious,
    max_samples_per_client=args.max_samples_per_client,
    seed=int(config.get("seed", 42)),
    init_checkpoint=args.init_checkpoint,
    on_progress=report_progress,
)

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print()
print(f"{'scenario':<15} {'mode':<18} {'macro_f1':>10} {'accuracy':>10}")
for row in report["results"]:
    print(f"{row['scenario']:<15} {row['mode']:<18} {row['test_macro_f1']:>10.4f} {row['test_accuracy']:>10.4f}")

print()
print("detector (pca_cluster_ema) TPR/FPR/TNR across attacked scenarios:", report["detection_metrics"])
print("mean aggregation overhead per mode (seconds):", report["aggregation_overhead_seconds"])
print(f"Saved: {output_path.resolve()}")
