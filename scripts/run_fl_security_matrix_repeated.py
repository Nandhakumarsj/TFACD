"""Repeat the existing FL security matrix over fixed seeds and summarize CIs.

This is orchestration around run_fl_security_matrix.py, not a separate FL
implementation. Per-seed stdout/stderr and JSON outputs are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import t as student_t

MODES = ("fedavg", "fedprox", "ftil")
ATTACKS = ("label_flip", "sign_flip", "gaussian_noise", "gradual_scaling")
METRIC_KEYS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "attack_detection_rate", "attack_detection_precision",
)

parser = argparse.ArgumentParser(description="Repeat the FTIL/FedAvg/FedProx security matrix over several seeds.")
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--partitions-dir", default="artifacts/data/partitions_10")
parser.add_argument("--num-clients", type=int, default=10)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--proximal-mu", type=float, default=0.01)
parser.add_argument("--max-samples-per-client", type=int, default=15000)
parser.add_argument("--init-checkpoint", default=None)
parser.add_argument("--seeds", nargs="+", type=int, required=True)
parser.add_argument("--attacks", nargs="+", choices=ATTACKS, default=list(ATTACKS))
parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
parser.add_argument("--output-dir", default="artifacts/second_review/fl_repeated")
args = parser.parse_args()


def ci(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if not n:
        return {"n_seeds": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(statistics.mean(values))
    if n < 2:
        return {"n_seeds": n, "mean": mean, "std": None, "ci95_low": None, "ci95_high": None}
    std = float(statistics.stdev(values))
    half = float(student_t.ppf(0.975, n - 1) * std / math.sqrt(n))
    return {"n_seeds": n, "mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
script_path = Path(__file__).with_name("run_fl_security_matrix.py").resolve()
seed_reports: list[dict[str, Any]] = []

for seed in args.seeds:
    output_path = out_dir / f"fl_security_matrix_seed_{seed}.json"
    log_path = out_dir / f"fl_security_matrix_seed_{seed}.log"
    cmd = [
        sys.executable, str(script_path),
        "--config", args.config,
        "--partitions-dir", args.partitions_dir,
        "--num-clients", str(args.num_clients),
        "--rounds", str(args.rounds),
        "--proximal-mu", str(args.proximal_mu),
        "--max-samples-per-client", str(args.max_samples_per_client),
        "--seed", str(seed),
        "--attacks", *args.attacks,
        "--modes", *args.modes,
        "--output", str(output_path),
    ]
    if args.init_checkpoint:
        cmd.extend(["--init-checkpoint", args.init_checkpoint])
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log_text = "COMMAND: " + json.dumps(cmd) + "\n\nSTDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr
    log_path.write_text(log_text, encoding="utf-8")
    seed_reports.append({
        "seed": seed,
        "command": cmd,
        "exit_status": completed.returncode,
        "json_path": str(output_path),
        "log_path": str(log_path),
    })
    print(f"seed={seed}: exit_status={completed.returncode}; log={log_path}")
    if completed.returncode != 0:
        raise SystemExit(f"FL matrix failed for seed {seed}; inspect {log_path}. Other seed reports are preserved.")

per_seed_data: dict[int, dict[str, Any]] = {}
for report in seed_reports:
    per_seed_data[report["seed"]] = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))

# Fail closed on a pairing-contract violation. A seed-level JSON must use one
# initialization and one partition manifest, and every method in a given
# attack/fraction cell must be exposed to the same malicious client identities.
pairing_checks: dict[str, Any] = {}
for seed, data in per_seed_data.items():
    if data.get("seed") != seed:
        raise SystemExit(f"Seed metadata mismatch: wrapper seed={seed}, child seed={data.get('seed')}")
    by_cell: dict[tuple[str, float], dict[str, list[int]]] = defaultdict(dict)
    for row in data.get("results", []):
        if row["attack"] == "none":
            continue
        cell = (str(row["attack"]), float(row["malicious_fraction_requested"]))
        mode = str(row["mode"])
        ids = sorted(int(cid) for cid in row.get("malicious_clients", []))
        if mode in by_cell[cell] and by_cell[cell][mode] != ids:
            raise SystemExit(f"Duplicate cell has inconsistent malicious IDs for seed={seed}, cell={cell}, mode={mode}")
        by_cell[cell][mode] = ids
    for cell, methods in by_cell.items():
        distinct_id_sets = {tuple(ids) for ids in methods.values()}
        if len(distinct_id_sets) > 1:
            raise SystemExit(f"Unpaired malicious-client IDs for seed={seed}, cell={cell}: {methods}")
    pairing_checks[str(seed)] = {
        "seed_matches_child_metadata": True,
        "initial_state_sha256": data.get("initial_state_sha256"),
        "partition_manifest_sha256": data.get("partition_manifest_sha256"),
        "malicious_ids_match_across_methods": True,
        "attack_fraction_cells_checked": len(by_cell),
    }

cells: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
for seed, data in per_seed_data.items():
    for row in data["results"]:
        key = (row["mode"], row["attack"], float(row["malicious_fraction_requested"]))
        cells[key].append({"seed": seed, **row})

summaries: list[dict[str, Any]] = []
for (mode, attack, fraction), runs in sorted(cells.items()):
    summary: dict[str, Any] = {
        "mode": mode,
        "attack": attack,
        "malicious_fraction_requested": fraction,
        "n_seeds": len(runs),
        "metrics": {},
    }
    for metric in METRIC_KEYS:
        summary["metrics"][metric] = ci([float(r["metrics"][metric]) for r in runs])
    for metric in (
        "malicious_client_detection_rate", "benign_client_false_positive_rate",
        "global_model_degradation_macro_f1", "global_model_degradation_accuracy",
        "runtime_seconds",
    ):
        values = [float(r[metric]) for r in runs if r.get(metric) is not None]
        summary[metric] = ci(values) if values else None
    summary["malicious_fraction_observed"] = ci([float(r["effective_malicious_fraction"]) for r in runs])
    summary["malicious_client_ids_by_seed"] = {str(r["seed"]): r["malicious_clients"] for r in runs}
    counts = [r.get("client_level_confusion") for r in runs if r.get("client_level_confusion")]
    summary["client_level_confusion"] = (
        {key: sum(int(c[key]) for c in counts) for key in ("tp", "fn", "fp", "tn")} if counts else None
    )
    diag = [d for r in runs for d in r.get("attack_diagnostics", [])]
    summary["gaussian_noise_diagnostics"] = {
        "client_round_records": len(diag),
        "mean_clean_update_l2_norm": float(np.mean([d["clean_update_l2_norm"] for d in diag])) if diag else None,
        "mean_injected_noise_l2_norm": float(np.mean([d["injected_noise_l2_norm"] for d in diag])) if diag else None,
        "mean_observed_noise_to_update_norm_ratio": float(np.mean([d["observed_noise_to_update_norm_ratio"] for d in diag if d["observed_noise_to_update_norm_ratio"] is not None])) if any(d["observed_noise_to_update_norm_ratio"] is not None for d in diag) else None,
        "per_client_round": diag,
    }
    summaries.append(summary)

# Paired method contrasts use the same seed, initialization, client partitions,
# local data-order seed, attack and malicious client IDs for each comparison.
lookup = {
    (seed, r["mode"], r["attack"], float(r["malicious_fraction_requested"])): r
    for seed, data in per_seed_data.items() for r in data["results"]
}
paired: list[dict[str, Any]] = []
for attack in ("none", *args.attacks):
    fractions = [0.0] if attack == "none" else [0.1, 0.2, 0.3]
    for fraction in fractions:
        for other in ("fedavg", "fedprox"):
            if other not in args.modes or "ftil" not in args.modes:
                continue
            for metric in ("accuracy", "macro_f1"):
                diffs = []
                for seed in args.seeds:
                    left = lookup.get((seed, "ftil", attack, fraction))
                    right = lookup.get((seed, other, attack, fraction))
                    if left is not None and right is not None:
                        diffs.append(float(left["metrics"][metric]) - float(right["metrics"][metric]))
                paired.append({
                    "attack": attack, "malicious_fraction_requested": fraction,
                    "contrast": f"ftil_minus_{other}", "metric": metric, **ci(diffs),
                })

report = {
    "experiment": "Repeated FTIL/FedAvg/FedProx security matrix",
    "seeds": args.seeds,
    "num_clients": args.num_clients,
    "rounds": args.rounds,
    "attacks": args.attacks,
    "modes": args.modes,
    "proximal_mu": args.proximal_mu,
    "partition_manifest_sha256": file_sha(Path(args.partitions_dir) / "manifest.json") if (Path(args.partitions_dir) / "manifest.json").exists() else None,
    "config_sha256": file_sha(args.config),
    "initialization_by_seed": {str(s): d.get("initial_state_sha256") for s, d in per_seed_data.items()},
    "pairing_checks": pairing_checks,
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "metric_note": "Record-level IDS attack detection metrics are separate from client-level FTIL detection metrics.",
    "seed_runs": seed_reports,
    "summaries": summaries,
    "paired_method_differences": paired,
}
summary_path = out_dir / "fl_security_matrix_repeated_summary.json"
summary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(f"Saved repeated-run summary: {summary_path.resolve()}")
