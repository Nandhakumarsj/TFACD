from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import scipy
from scipy.stats import t as student_t

from tfacd.common.config import load_config
from tfacd.evaluation.astb_benchmark import (
    ABLATIONS,
    COMPONENT_ABLATIONS,
    BenchmarkScenario,
    EvaluationPolicyAdapter,
    SecurityPolicyOracle,
    generate_scenarios,
    load_oracle_config,
    load_prediction_policy_config,
    load_threat_mapping,
    run_ablation,
)

parser = argparse.ArgumentParser(
    description="Run reproducible coarse and component-wise ASTB ablations."
)
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--oracle", default="configs/astb_oracle.yaml")
parser.add_argument("--predictor-policy", default="configs/astb_predictor.yaml",
                    help="Prediction-side RBAC and trust thresholds; separate from oracle labels.")
parser.add_argument("--dataset", default="artifacts/astb_benchmark/astb_iiot_security_evaluation.jsonl")
parser.add_argument("--output-dir", default="artifacts/second_review/astb")
parser.add_argument(
    "--seeds", nargs="+", type=int, default=None,
    help="Generate a paired scenario set for each seed (e.g. --seeds 42 43 44 45 46). "
         "Every variant sees the same scenarios within each seed.",
)
parser.add_argument("--count", type=int, default=1000, help="Scenarios per seed when --seeds is supplied.")
args = parser.parse_args()


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ci_summary(values: list[float]) -> dict[str, float | int | None]:
    """Mean, sample SD and two-sided 95% Student-t CI across independent seeds."""
    n = len(values)
    if not n:
        return {"n_seeds": 0, "mean": 0.0, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(statistics.mean(values))
    if n < 2:
        return {"n_seeds": n, "mean": mean, "std": None, "ci95_low": None, "ci95_high": None}
    std = float(statistics.stdev(values))
    half = float(student_t.ppf(0.975, n - 1) * std / math.sqrt(n))
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


BOUNDED_METRIC_KEYS = frozenset({
    "accuracy", "macro_precision", "macro_recall", "macro_f1", "precision", "recall", "f1",
    "false_authorization_rate", "far_unsafe", "false_allow_rate",
    "unsafe_action_blocking_rate", "unsafe_human_review_rate",
    "safe_action_preservation_rate", "non_unsafe_allow_rate", "non_unsafe_human_review_rate", "legitimate_human_review_rate",
    "human_review_precision", "human_review_recall",
    "oracle_allow_preservation_rate", "oracle_allow_human_review_rate",
    "oracle_non_allow_false_authorization_rate", "prompt_injection_success_rate",
    "unauthorized_tool_call_rate", "policy_violation_allow_rate", "human_review_rate",
    "context_consistency_raw_mean", "context_consistency_raw_min", "context_consistency_raw_max",
})


def aggregate_metric_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric metrics over seeds; sum count matrices, never average them."""
    if not dicts:
        return {}
    out: dict[str, Any] = {}
    keys = set.intersection(*(set(d) for d in dicts))
    out["confidence_intervals"] = {}
    for key in sorted(keys):
        vals = [d[key] for d in dicts]
        if key in {"confusion_matrix", "safety_confusion_matrix"}:
            out[key] = np.sum(np.asarray(vals, dtype=np.int64), axis=0).tolist()
        elif key in {"by_attack_category", "by_attack_subtype", "per_class"}:
            categories = sorted(set().union(*(d[key].keys() for d in dicts)))
            out[key] = {
                category: aggregate_metric_dicts([d[key][category] for d in dicts if category in d[key]])
                for category in categories
            }
        elif isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
            summary = ci_summary([float(v) for v in vals])
            is_count = (
                key in {"n", "unsafe_count", "legitimate_count"}
                or key.endswith(("_numerator", "_denominator", "_count"))
            )
            # Counts describe the complete pooled dataset. Rates/latencies are
            # means over repeat seeds; their variability is available separately.
            if is_count:
                out[key] = int(sum(int(v) for v in vals))
            else:
                summary = ci_summary([float(v) for v in vals])
                out[key] = summary["mean"]
                if key in BOUNDED_METRIC_KEYS:
                    if summary["ci95_low"] is not None:
                        summary["ci95_low"] = max(0.0, float(summary["ci95_low"]))
                    if summary["ci95_high"] is not None:
                        summary["ci95_high"] = min(1.0, float(summary["ci95_high"]))
                out["confidence_intervals"][key] = summary
        else:
            out[key] = vals[0]
    return out


def load_existing_dataset(path: Path, oracle_hash: str) -> tuple[list[BenchmarkScenario], int | None]:
    manifest_path = path.with_name("dataset_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    seed = manifest.get("seed")
    scenarios: list[BenchmarkScenario] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        payload.setdefault("benchmark_seed", seed if seed is not None else 42)
        payload.setdefault("oracle_version", str(manifest.get("oracle_version", "unversioned")))
        payload.setdefault("oracle_sha256", manifest.get("oracle_sha256", oracle_hash))
        scenarios.append(BenchmarkScenario(**payload))
    if not scenarios:
        raise SystemExit(f"No scenarios found in {path}")
    return scenarios, int(seed) if seed is not None else None


config = load_config(args.config)
policy_path = config["runtime"]["trust_policy_path"]
threat_mapping_path = config["runtime"]["threat_context_mapping"]
policy = load_config(policy_path)
threat_mapping = load_threat_mapping(threat_mapping_path)
oracle_cfg = load_oracle_config(args.oracle)
oracle = SecurityPolicyOracle(oracle_cfg, policy, threat_mapping)
oracle.validate()
oracle_hash = sha256(args.oracle)
predictor_cfg = load_prediction_policy_config(args.predictor_policy)
predictor_policy = EvaluationPolicyAdapter(predictor_cfg, policy)
predictor_policy.validate()
predictor_policy_hash = sha256(args.predictor_policy)

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)

runs: list[dict[str, Any]] = []
all_scenarios: list[BenchmarkScenario] = []
if args.seeds:
    if args.count < 100:
        raise SystemExit("--count must be at least 100")
    for seed in args.seeds:
        scenarios = generate_scenarios(
            args.count, seed=seed, oracle=oracle, oracle_sha256=oracle_hash
        )
        seed_path = out / "seed_datasets" / f"seed_{seed}.jsonl"
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(
            "".join(json.dumps(s.to_dict(), sort_keys=True) + "\n" for s in scenarios),
            encoding="utf-8",
        )
        all_scenarios.extend(scenarios)
        runs.append({"seed": seed, "scenarios": scenarios})
else:
    dataset_path = Path(args.dataset)
    scenarios, seed = load_existing_dataset(dataset_path, oracle_hash)
    all_scenarios.extend(scenarios)
    runs.append({"seed": seed, "scenarios": scenarios, "dataset": str(dataset_path)})

matrix_specs = (("coarse", ABLATIONS), ("component", COMPONENT_ABLATIONS))
all_summaries: dict[str, Any] = {}
manifest = {
    "experiment": "ASTB-IIoT paired ablation evaluation",
    "oracle_version": str(oracle_cfg.get("oracle_version", "unversioned")),
    "oracle_sha256": oracle_hash,
    "predictor_policy_version": str(predictor_cfg.get("predictor_policy_version", "unversioned")),
    "predictor_policy_sha256": predictor_policy_hash,
    "prediction_thresholds": {"block": predictor_policy.block_threshold,
                             "human_review_below": predictor_policy.review_threshold,
                             "context_consistency": predictor_policy.context_consistency_threshold},
    "policy_sha256": sha256(policy_path),
    "threat_mapping_sha256": sha256(threat_mapping_path),
    "config_sha256": sha256(args.config),
    "astb_engine_sha256": sha256("src/tfacd/evaluation/astb_benchmark.py"),
    "ablation_runner_sha256": sha256(__file__),
    "dataset_generator_sha256": sha256("scripts/generate_astb_benchmark.py"),
    "context_consistency_threshold": oracle.context_consistency_threshold,
    "seeds": [r["seed"] for r in runs],
    "scenario_count_per_seed": len(runs[0]["scenarios"]),
    "run_count": len(runs),
    "paired_design": "Every ablation variant evaluates the same scenario records for a given seed.",
    "primary_matrix": "component",
    "matrix_purposes": {
        "component": "Incremental component study: baseline, semantic+policy, add context, add behaviour, full ASTB.",
        "coarse": "Separate coarse comparison: agent-only, static rules, semantic check, full ASTB; not pooled with component matrix.",
    },
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "scikit_learn_version": sklearn.__version__,
    "scipy_version": scipy.__version__,
    "command_argv": sys.argv,
    "metric_definitions": {
        "UABR": "unsafe=true and predicted BLOCK / all unsafe=true scenarios",
        "FAR": "unsafe=true and predicted ALLOW / all unsafe=true scenarios",
        "FRR": "unsafe=false and predicted BLOCK / all unsafe=false scenarios; HUMAN_REVIEW is not counted as BLOCK",
        "FRR_interpretation": "hard-block rejection rate for legitimate proposals; report legitimate_human_review_rate separately",
        "unsafe_human_review_rate": "unsafe=true and predicted HUMAN_REVIEW / all unsafe=true scenarios",
        "SAPR": "expected_decision=ALLOW and predicted_decision=ALLOW / all expected_decision=ALLOW scenarios",
        "confusion_matrix": "three-class oracle-decision matrix: rows=expected_decision and columns=predicted_decision in decision_labels order",
        "safety_confusion_matrix": "two-row by three-column matrix: rows=unsafe/legitimate, columns=ALLOW/BLOCK/HUMAN_REVIEW",
        "count_aggregation": "numerator/denominator/count fields are summed across seeds; rates and latencies are mean seed-level values with confidence intervals",
        "non_unsafe_allow_rate": "unsafe=false and predicted ALLOW / all unsafe=false scenarios; may include HUMAN_REVIEW controls",
        "human_review_precision": "expected HUMAN_REVIEW among all predicted HUMAN_REVIEW scenarios",
        "human_review_recall": "predicted HUMAN_REVIEW among all expected HUMAN_REVIEW scenarios",
        "oracle_non_allow_false_authorization_rate": "expected BLOCK or HUMAN_REVIEW predicted ALLOW / all expected non-ALLOW scenarios",
        "oracle_allow_preservation_rate": "oracle ALLOW and predicted ALLOW / all oracle ALLOW scenarios",
        "human_review_rate": "predicted HUMAN_REVIEW / all scenarios",
        "confidence_interval": "two-sided 95% Student-t interval across five seed-level metrics; rows are paired by seed for variant differences",
    },
}

for matrix_name, specs in matrix_specs:
    seed_summaries: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for run in runs:
        rows, summary = run_ablation(
            run["scenarios"], specs=specs, predictor_policy=predictor_policy, threat_mapping=threat_mapping
        )
        for row in rows:
            row["run_seed"] = run["seed"]
        combined_rows.extend(rows)
        seed_summaries.append({"seed": run["seed"], "variants": summary["variants"],
                               "oracle_label_distribution": summary["oracle_label_distribution"]})

    variant_names = [spec.name for spec in specs]
    variants = {
        name: aggregate_metric_dicts([entry["variants"][name] for entry in seed_summaries])
        for name in variant_names
    }
    paired_comparisons: dict[str, Any] = {}
    if "full_astb" in variant_names:
        primary_metrics = [
            "accuracy", "macro_f1", "far_unsafe", "unsafe_action_blocking_rate",
            "false_rejection_rate", "safe_action_preservation_rate",
            "oracle_allow_preservation_rate", "human_review_rate",
        ]
        for other in variant_names:
            if other == "full_astb":
                continue
            paired_comparisons[other] = {}
            for metric in primary_metrics:
                diffs = [
                    float(entry["variants"]["full_astb"][metric]) - float(entry["variants"][other][metric])
                    for entry in seed_summaries
                ]
                difference_summary = ci_summary(diffs)
                # Differences of bounded rates/scores lie in [-1, 1].
                if difference_summary["ci95_low"] is not None:
                    difference_summary["ci95_low"] = max(-1.0, float(difference_summary["ci95_low"]))
                if difference_summary["ci95_high"] is not None:
                    difference_summary["ci95_high"] = min(1.0, float(difference_summary["ci95_high"]))
                paired_comparisons[other][metric] = difference_summary

    summary_out = {
        "matrix": matrix_name,
        "variants": variants,
        "variant_components": {
            spec.name: {
                "semantic": spec.semantic, "context": spec.context, "behavior": spec.behavior,
                "history": spec.history, "policy": spec.policy, "dynamic_trust": spec.dynamic_trust,
            }
            for spec in specs
        },
        "per_seed": seed_summaries,
        "paired_full_astb_differences": paired_comparisons,
        "run_count": len(runs),
        "seeds": [run["seed"] for run in runs],
    }
    all_summaries[matrix_name] = summary_out
    (out / f"{matrix_name}_ablation_summary.json").write_text(
        json.dumps(summary_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    csv_path = out / f"{matrix_name}_ablation_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(combined_rows[0]) if combined_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_rows:
            serialized = dict(row)
            serialized["component_configuration"] = json.dumps(row["component_configuration"], sort_keys=True)
            for key, value in list(serialized.items()):
                if isinstance(value, (dict, list)) and key != "component_configuration":
                    serialized[key] = json.dumps(value, sort_keys=True)
            writer.writerow(serialized)

    print(f"\n=== {matrix_name.upper()} ABLATION ({len(runs)} run(s)) ===")
    print(f"{'variant':<24} {'F1':>8} {'FAR':>8} {'UABR':>8} {'FRR':>8} {'SAPR':>8} {'Review':>8} {'p95 ms':>10}")
    for name in variant_names:
        metric = variants[name]
        print(
            f"{name:<24} {metric['macro_f1']:>8.3f} {metric['far_unsafe']:>8.3f} "
            f"{metric['unsafe_action_blocking_rate']:>8.3f} {metric['false_rejection_rate']:>8.3f} "
            f"{metric['safe_action_preservation_rate']:>8.3f} "
            f"{metric['human_review_rate']:>8.3f} {metric['latency_p95_ms']:>10.2f}"
        )
        if len(runs) > 1:
            ci = metric.get("confidence_intervals", {}).get("far_unsafe")
            if ci:
                print(f"  FAR mean={ci['mean']:.3f}, SD={ci['std']:.3f}, 95% CI=[{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}]")

manifest["matrices"] = {
    name: {
        "variant_names": list(summary["variant_components"]),
        "variant_components": summary["variant_components"],
        "rows": sum(len(run["scenarios"]) * len(summary["variants"]) for run in runs),
    }
    for name, summary in all_summaries.items()
}
manifest_path = out / "experiment_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print(f"\nSaved experiment manifest: {manifest_path.resolve()}")
print(f"Saved summaries and scenario-level CSV rows under: {out.resolve()}")
