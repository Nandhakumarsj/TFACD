#!/usr/bin/env python3
"""Generate report-ready TFACD ASTB/FL figures from JSON/CSV experiment outputs.

Default inputs are the canonical output paths of scripts/run_astb_ablation.py and
scripts/run_fl_security_matrix.py. For repeated FL runs, supply every seed-level
JSON via --fl-jsons; confidence intervals are calculated across those input files.
The script never generates or imputes experimental values and will skip absent inputs.

Example (after a fresh experiment):
  python scripts/plot_second_review_results.py \
    --astb-summary artifacts/second_review/astb/component_ablation_summary.json \
    --astb-rows artifacts/second_review/astb/component_ablation_rows.csv \
    --fl-jsons artifacts/second_review/fl_seed_42.json \
                artifacts/second_review/fl_seed_43.json \
                artifacts/second_review/fl_seed_44.json \
                artifacts/second_review/fl_seed_45.json \
                artifacts/second_review/fl_seed_46.json \
    --output-dir artifacts/second_review_figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.stats import t as student_t
except ImportError:  # ASTB experiment runner itself requires SciPy; keep plotting diagnostic usable.
    scipy = None
    student_t = None


RATE_METRICS = {
    "FAR (unsafe → ALLOW)": ("far_unsafe", "false_allow_rate"),
    "UABR (unsafe → BLOCK)": ("unsafe_action_blocking_rate",),
    "SAPR (legitimate → ALLOW)": ("safe_action_preservation_rate",),
    "Oracle ALLOW preservation": ("oracle_allow_preservation_rate",),
    "Human-review rate": ("human_review_rate",),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.") or "figure"


def metric_value(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def ci_bounds(metrics: dict[str, Any], key: str) -> tuple[float, float] | None:
    ci = metrics.get("confidence_intervals", {}).get(key)
    if not isinstance(ci, dict):
        return None
    low, high = ci.get("ci95_low"), ci.get("ci95_high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return None
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        return None
    # These intervals are used by the rate plots; rates live in [0,1]. The
    # experiment JSON retains the original Student-t bounds, while the figure
    # clips visually impossible endpoints to the metric's natural range.
    return max(0.0, float(low)), min(1.0, float(high))


def error_sizes(mean: float, bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    if bounds is None:
        return None
    low, high = bounds
    return max(0.0, mean - low), max(0.0, high - mean)


def save_figure(fig: plt.Figure, out: Path, name: str) -> str:
    target = out / name
    fig.savefig(target, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target.name


def plot_astb_rates(summary: dict[str, Any], out: Path) -> list[str]:
    variants = summary.get("variants", {})
    names = list(variants)
    if not names:
        return []
    files: list[str] = []
    groups = [
        (
            "astb_component_security_rates.png",
            "ASTB ablation: unsafe blocking and legitimate-action preservation",
            {
                "FAR (unsafe → ALLOW)": ("far_unsafe", "false_allow_rate"),
                "UABR (unsafe → BLOCK)": ("unsafe_action_blocking_rate",),
                "SAPR (legitimate → ALLOW)": ("safe_action_preservation_rate",),
            },
        ),
        (
            "astb_component_review_and_preservation.png",
            "ASTB ablation: authorized-action preservation and review burden",
            {
                "Oracle ALLOW preservation": ("oracle_allow_preservation_rate",),
                "Human-review rate": ("human_review_rate",),
            },
        ),
    ]
    for filename, title, measures in groups:
        fig, ax = plt.subplots(figsize=(max(9.4, len(names) * 1.75), 5.8))
        x = np.arange(len(names), dtype=float)
        metric_count = len(measures)
        width = min(0.22, 0.64 / metric_count)
        offsets = (np.arange(metric_count) - (metric_count - 1) / 2.0) * width
        for j, (label, keys) in enumerate(measures.items()):
            vals: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            have_any_ci = False
            for name in names:
                metric = variants[name]
                value = metric_value(metric, keys)
                vals.append(float("nan") if value is None else value * 100.0)
                key_used = next((k for k in keys if k in metric), keys[0])
                es = error_sizes(value, ci_bounds(metric, key_used)) if value is not None else None
                if es is None:
                    lows.append(0.0)
                    highs.append(0.0)
                else:
                    have_any_ci = True
                    lows.append(es[0] * 100.0)
                    highs.append(es[1] * 100.0)
            positions = x + offsets[j]
            ax.bar(positions, vals, width=width * 0.9, label=label, alpha=0.88)
            if have_any_ci:
                ax.errorbar(positions, vals, yerr=np.array([lows, highs]), fmt="none", capsize=3, linewidth=1.0)
        ax.set_xticks(x, names, rotation=16, ha="right")
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(ncols=metric_count, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        fig.tight_layout()
        files.append(save_figure(fig, out, filename))
    return files


def plot_astb_confusion_matrices(summary: dict[str, Any], out: Path) -> list[str]:
    variants = summary.get("variants", {})
    labels = summary.get("decision_labels", ["ALLOW", "BLOCK", "HUMAN_REVIEW"])
    files: list[str] = []
    for name, metrics in variants.items():
        cm = metrics.get("confusion_matrix")
        if not isinstance(cm, list) or len(cm) != len(labels):
            continue
        matrix = np.asarray(cm, dtype=float)
        if matrix.shape != (len(labels), len(labels)):
            continue
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        image = ax.imshow(matrix, cmap="Blues", aspect="auto")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Scenario count")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xlabel("Predicted decision")
        ax.set_ylabel("Oracle reference label")
        ax.set_title(f"ASTB confusion matrix — {name}")
        threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "black", fontsize=10)
        fig.tight_layout()
        files.append(save_figure(fig, out, f"astb_confusion_{safe_name(name)}.png"))
    return files


def plot_astb_attack_categories(summary: dict[str, Any], out: Path) -> list[str]:
    variants = summary.get("variants", {})
    if not variants:
        return []
    first = next(iter(variants.values()))
    categories = sorted(first.get("by_attack_category", {}).keys())
    names = list(variants)
    if not categories:
        return []
    matrix = np.full((len(categories), len(names)), np.nan, dtype=float)
    for j, name in enumerate(names):
        by_category = variants[name].get("by_attack_category", {})
        for i, cat in enumerate(categories):
            m = by_category.get(cat, {})
            val = metric_value(m, ("macro_f1",))
            if val is not None:
                matrix[i, j] = val * 100.0
    fig, ax = plt.subplots(figsize=(max(9.2, len(names) * 1.4), max(4.4, len(categories) * 0.62)))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Macro-F1 (%)")
    ax.set_xticks(np.arange(len(names)), names, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(categories)), categories)
    ax.set_xlabel("Ablation variant")
    ax.set_ylabel("Scenario category")
    ax.set_title("ASTB macro-F1 by attack/control category")
    for i in range(len(categories)):
        for j in range(len(names)):
            if math.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                        color="white" if matrix[i, j] > 55 else "black", fontsize=8)
    fig.tight_layout()
    return [save_figure(fig, out, "astb_macro_f1_by_category.png")]


def plot_astb_latency(summary: dict[str, Any], out: Path) -> list[str]:
    variants = summary.get("variants", {})
    names = list(variants)
    if not names:
        return []
    means = [metric_value(variants[n], ("latency_mean_ms",)) for n in names]
    p95s = [metric_value(variants[n], ("latency_p95_ms",)) for n in names]
    if all(v is None for v in means + p95s):
        return []
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(9.0, len(names) * 1.5), 5.2))
    width = 0.35
    ax.bar(x - width / 2, [np.nan if v is None else v for v in means], width, label="Mean")
    ax.bar(x + width / 2, [np.nan if v is None else v for v in p95s], width, label="95th percentile")
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("ASTB prediction latency by ablation variant")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    return [save_figure(fig, out, "astb_latency.png")]


def plot_astb_provenance(rows_path: Path, out: Path) -> tuple[list[str], dict[str, Any]]:
    if not rows_path.is_file():
        return [], {"available": False, "reason": f"CSV not found: {rows_path}"}
    df = pd.read_csv(rows_path)
    required = {"variant", "decision_stage", "decision_reason", "policy_status", "semantic_status",
                "context_status", "behavior_status", "history_status", "dynamic_trust_status"}
    missing = sorted(required - set(df.columns))
    if missing:
        return [], {"available": False, "reason": "CSV predates provenance fields", "missing_columns": missing}
    files: list[str] = []
    full = df[df["variant"] == "full_astb"]
    if full.empty:
        full = df
    component_cols = ["policy_status", "semantic_status", "context_status", "behavior_status", "history_status", "dynamic_trust_status"]
    colors = None
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    bottom = np.zeros(len(component_cols), dtype=float)
    statuses = ["disabled", "not_reached", "passed", "flagged", "blocked", "human_review", "verified_allow"]
    # Count status values by gate across all scenario rows of the full model.
    table = pd.DataFrame(index=component_cols)
    for status in statuses:
        table[status] = [int((full[c].astype(str) == status).sum()) for c in component_cols]
    for status in statuses:
        vals = table[status].to_numpy(dtype=float)
        if vals.sum() == 0:
            continue
        ax.barh(np.arange(len(component_cols)), vals, left=bottom, label=status)
        bottom += vals
    ax.set_yticks(np.arange(len(component_cols)), [c.replace("_status", "").replace("_", " ").title() for c in component_cols])
    ax.set_xlabel("Scenario count (full_astb rows)")
    ax.set_title("ASTB component execution provenance — full variant")
    ax.invert_yaxis()
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    files.append(save_figure(fig, out, "astb_component_execution_provenance.png"))

    counts = Counter(full["decision_reason"].fillna("missing").astype(str))
    top = counts.most_common(12)
    if top:
        labels = [x[0] for x in top][::-1]
        values = [x[1] for x in top][::-1]
        fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.42 * len(labels))))
        ax.barh(labels, values)
        ax.set_xlabel("Scenario count")
        ax.set_title("ASTB final decision reasons — full variant")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        files.append(save_figure(fig, out, "astb_decision_reason_counts.png"))
    meta = {
        "available": True,
        "rows_loaded": int(len(df)),
        "full_variant_rows": int(len(full)),
        "decision_reason_counts": dict(counts),
    }
    return files, meta


def plot_astb_paired_deltas(summary: dict[str, Any], out: Path) -> list[str]:
    diffs = summary.get("paired_full_astb_differences", {})
    metrics = [
        "macro_f1", "far_unsafe", "unsafe_action_blocking_rate",
        "safe_action_preservation_rate", "oracle_allow_preservation_rate", "human_review_rate",
    ]
    variants = [v for v in diffs if isinstance(diffs[v], dict)]
    if not variants:
        return []
    files: list[str] = []
    for metric in metrics:
        entries = []
        for variant in variants:
            item = diffs[variant].get(metric)
            if not isinstance(item, dict) or not isinstance(item.get("mean"), (int, float)):
                continue
            mean = float(item["mean"])
            low, high = item.get("ci95_low"), item.get("ci95_high")
            err = None
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                err = (max(0.0, mean - float(low)), max(0.0, float(high) - mean))
            entries.append((variant, mean, err))
        if not entries:
            continue
        labels = [f"Full ASTB − {v}" for v, _, _ in entries][::-1]
        values = [m for _, m, _ in entries][::-1]
        errors = [e for _, _, e in entries][::-1]
        y = np.arange(len(entries), dtype=float)
        fig, ax = plt.subplots(figsize=(8.8, max(3.6, 0.62 * len(entries))))
        for yi, mean, err in zip(y, values, errors):
            xerr = None if err is None else np.array([[err[0]], [err[1]]])
            ax.errorbar(mean, yi, xerr=xerr, fmt="o", capsize=3, linewidth=1.2)
        ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Paired mean difference (95% CI; full ASTB minus comparator)")
        ax.set_title(f"Paired effect on {metric.replace('_', ' ')}")
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        fig.tight_layout()
        files.append(save_figure(fig, out, f"astb_paired_delta_{safe_name(metric)}.png"))
    return files


def load_fl_files(paths: Iterable[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    file_meta: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            file_meta.append({"path": str(path), "available": False, "reason": "not found"})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        file_meta.append({
            "path": str(path), "available": True, "sha256": sha256(path),
            "seed": data.get("seed"), "experiment": data.get("experiment"),
            "results_rows": len(data.get("results", [])),
        })
        for item in data.get("results", []):
            metrics = item.get("metrics") or {}
            rows.append({
                "input_file": str(path),
                "seed": item.get("training_seed", data.get("seed")),
                "seed_count": 1,
                "mode": str(item.get("mode", "unknown")),
                "attack": str(item.get("attack", "unknown")),
                "fraction": float(item.get("malicious_fraction_requested", item.get("effective_malicious_fraction", 0.0))) * 100.0,
                "effective_fraction": float(item.get("effective_malicious_fraction", item.get("malicious_fraction_requested", 0.0))) * 100.0,
                "macro_f1": metrics.get("macro_f1"),
                "macro_f1_ci_low": None,
                "macro_f1_ci_high": None,
                "accuracy": metrics.get("accuracy"),
                "attack_detection_rate": metrics.get("attack_detection_rate"),
                "malicious_client_detection_rate": item.get("malicious_client_detection_rate"),
                "benign_client_false_positive_rate": item.get("benign_client_false_positive_rate"),
                "runtime_seconds": item.get("runtime_seconds"),
                "source_is_repeated_summary": False,
            })
        # The repeated-run wrapper emits summaries instead of a top-level `results`
        # array. Read its seed-level means/CIs directly when that file is supplied.
        if not data.get("results") and isinstance(data.get("summaries"), list):
            for item in data["summaries"]:
                metrics = item.get("metrics") or {}
                macro = metrics.get("macro_f1") or {}
                attack_detection = metrics.get("attack_detection_rate") or {}
                accuracy = metrics.get("accuracy") or {}
                malicious_detection = item.get("malicious_client_detection_rate") or {}
                benign_fpr = item.get("benign_client_false_positive_rate") or {}
                rows.append({
                    "input_file": str(path),
                    "seed": "repeated-summary",
                    "seed_count": int(macro.get("n_seeds", item.get("n_seeds", 0)) or 0),
                    "mode": str(item.get("mode", "unknown")),
                    "attack": str(item.get("attack", "unknown")),
                    "fraction": float(item.get("malicious_fraction_requested", 0.0)) * 100.0,
                    "effective_fraction": float((item.get("malicious_fraction_observed") or {}).get("mean", item.get("malicious_fraction_requested", 0.0))) * 100.0,
                    "macro_f1": macro.get("mean"),
                    "macro_f1_ci_low": macro.get("ci95_low"),
                    "macro_f1_ci_high": macro.get("ci95_high"),
                    "accuracy": accuracy.get("mean"),
                    "attack_detection_rate": attack_detection.get("mean"),
                    "malicious_client_detection_rate": malicious_detection.get("mean"),
                    "malicious_client_detection_ci_low": malicious_detection.get("ci95_low"),
                    "malicious_client_detection_ci_high": malicious_detection.get("ci95_high"),
                    "benign_client_false_positive_rate": benign_fpr.get("mean"),
                    "benign_client_fpr_ci_low": benign_fpr.get("ci95_low"),
                    "benign_client_fpr_ci_high": benign_fpr.get("ci95_high"),
                    "runtime_seconds": (item.get("runtime_seconds") or {}).get("mean"),
                    "source_is_repeated_summary": True,
                })
    return pd.DataFrame(rows), file_meta


def mean_ci(values: list[float]) -> tuple[float, float | None, float | None, int]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    n = len(values)
    if n == 0:
        return float("nan"), None, None, 0
    mean = float(statistics.mean(values))
    if n < 2 or student_t is None:
        return mean, None, None, n
    sd = float(statistics.stdev(values))
    half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return mean, mean - half, mean + half, n


def plot_fl_robustness(fl_df: pd.DataFrame, out: Path) -> list[str]:
    if fl_df.empty:
        return []
    files: list[str] = []
    modes = sorted(fl_df["mode"].dropna().astype(str).unique())
    for mode in modes:
        mode_df = fl_df[fl_df["mode"].astype(str) == mode]
        attack_names = [a for a in sorted(mode_df["attack"].dropna().astype(str).unique()) if a != "none"]
        for attack in attack_names:
            group = mode_df[mode_df["attack"].astype(str) == attack]
            controls = mode_df[(mode_df["attack"].astype(str) == "none") & (mode_df["fraction"] == 0)]
            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            plotted = False

            def add_fraction_point(fraction: float, cell: pd.DataFrame, *, use_precomputed_ci: bool = True) -> tuple[float, float] | None:
                nonlocal plotted
                values = pd.to_numeric(cell["macro_f1"], errors="coerce").dropna().tolist()
                if not values:
                    return None
                mean, low, high, _n = mean_ci(values)
                if use_precomputed_ci and len(cell) == 1:
                    row = cell.iloc[0]
                    if pd.notna(row.get("macro_f1_ci_low")) and pd.notna(row.get("macro_f1_ci_high")):
                        low, high = float(row["macro_f1_ci_low"]), float(row["macro_f1_ci_high"])
                yerr = None if low is None or high is None else np.array([[
                    max(0.0, mean - low) * 100.0], [max(0.0, high - mean) * 100.0]])
                ax.errorbar([fraction], [mean * 100.0], yerr=yerr, fmt="o", capsize=3)
                plotted = True
                return float(fraction), mean

            points: list[tuple[float, float]] = []
            for fraction, cell in controls.groupby("fraction"):
                point = add_fraction_point(float(fraction), cell)
                if point is not None:
                    points.append(point)
            for fraction, cell in group.groupby("fraction"):
                point = add_fraction_point(float(fraction), cell)
                if point is not None:
                    points.append(point)
            points.sort(key=lambda v: v[0])
            if plotted:
                ax.plot([p[0] for p in points], [p[1] * 100.0 for p in points], linewidth=1.2,
                        label=f"{mode} / {attack}")
                ax.set_xlabel("Requested malicious-client fraction (%)")
                ax.set_ylabel("IDS macro-F1 (%)")
                ax.set_ylim(0, 100)
                ax.set_title(f"FL robustness — {mode} under {attack} (0% = no-attack control)")
                ax.grid(alpha=0.25)
                ax.set_axisbelow(True)
                ax.legend(frameon=False)
                fig.tight_layout()
                files.append(save_figure(fig, out, f"fl_macro_f1_{safe_name(str(mode))}_{safe_name(str(attack))}.png"))
            else:
                plt.close(fig)

    # FTIL client-level detection and false positives are a different unit of analysis
    # from IDS record-level macro-F1. Plot separately, only where measured.
    for column, label, filename in [
        ("malicious_client_detection_rate", "Malicious-client detection rate (%)", "fl_ftil_malicious_client_detection.png"),
        ("benign_client_false_positive_rate", "Benign-client false-positive rate (%)", "fl_ftil_benign_client_false_positive.png"),
    ]:
        df = fl_df[fl_df[column].notna()].copy()
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.8, 5.1))
        for (mode, attack), group in df.groupby(["mode", "attack"], dropna=False):
            centers = []
            for fraction, cell in group.groupby("fraction"):
                vals = pd.to_numeric(cell[column], errors="coerce").dropna().tolist()
                if not vals:
                    continue
                mean, low, high, _n = mean_ci(vals)
                row = cell.iloc[0] if len(cell) == 1 else None
                low_key = "malicious_client_detection_ci_low" if column == "malicious_client_detection_rate" else "benign_client_fpr_ci_low"
                high_key = "malicious_client_detection_ci_high" if column == "malicious_client_detection_rate" else "benign_client_fpr_ci_high"
                if row is not None and pd.notna(row.get(low_key)) and pd.notna(row.get(high_key)):
                    low, high = float(row[low_key]), float(row[high_key])
                centers.append((float(fraction), mean, low, high))
            centers.sort(key=lambda v: v[0])
            if not centers:
                continue
            xs = [c[0] for c in centers]
            ys = [c[1] * 100.0 for c in centers]
            lows = [0.0 if c[2] is None else max(0.0, c[1] - c[2]) * 100.0 for c in centers]
            highs = [0.0 if c[3] is None else max(0.0, c[3] - c[1]) * 100.0 for c in centers]
            has_ci = any(c[2] is not None and c[3] is not None for c in centers)
            yerr = np.array([lows, highs]) if has_ci else None
            ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=2, label=f"{mode} / {attack}")
        ax.set_xlabel("Requested malicious-client fraction (%)")
        ax.set_ylabel(label)
        ax.set_ylim(0, 105)
        ax.set_title(label.replace(" (%)", "") + " — client-round analysis")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncols=2)
        fig.tight_layout()
        files.append(save_figure(fig, out, filename))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astb-summary", type=Path, default=Path("artifacts/second_review/astb/component_ablation_summary.json"))
    parser.add_argument("--astb-rows", type=Path, default=Path("artifacts/second_review/astb/component_ablation_rows.csv"))
    parser.add_argument("--fl-json", type=Path, default=Path("artifacts/second_review/fl_security_matrix.json"),
                        help="One FL matrix output JSON (used only if --fl-jsons is omitted).")
    parser.add_argument("--fl-jsons", type=Path, nargs="+", default=None,
                        help="All repeated seed-level FL matrix JSONs; CI is calculated across files.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/second_review_figures"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    input_manifest: dict[str, Any] = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "inputs": {}, "outputs": []}
    astb_loaded = False
    if args.astb_summary.is_file():
        summary = json.loads(args.astb_summary.read_text(encoding="utf-8"))
        input_manifest["inputs"]["astb_summary"] = {"path": str(args.astb_summary), "sha256": sha256(args.astb_summary)}
        astb_loaded = True
        generated += plot_astb_rates(summary, out)
        generated += plot_astb_confusion_matrices(summary, out)
        generated += plot_astb_attack_categories(summary, out)
        generated += plot_astb_latency(summary, out)
        generated += plot_astb_paired_deltas(summary, out)
    else:
        input_manifest["inputs"]["astb_summary"] = {"path": str(args.astb_summary), "available": False}

    provenance_files, provenance_meta = plot_astb_provenance(args.astb_rows, out)
    generated += provenance_files
    input_manifest["inputs"]["astb_rows"] = {
        "path": str(args.astb_rows), "available": args.astb_rows.is_file(),
        "sha256": sha256(args.astb_rows) if args.astb_rows.is_file() else None,
        "provenance_visualization": provenance_meta,
    }

    fl_paths = args.fl_jsons if args.fl_jsons else [args.fl_json]
    fl_df, fl_meta = load_fl_files(fl_paths)
    input_manifest["inputs"]["fl_jsons"] = fl_meta
    if not fl_df.empty:
        generated += plot_fl_robustness(fl_df, out)
        input_manifest["fl_observation_count"] = int(len(fl_df))
        if "seed_count" in fl_df and bool((pd.to_numeric(fl_df["seed_count"], errors="coerce") > 1).any()):
            input_manifest["fl_seed_count"] = int(pd.to_numeric(fl_df["seed_count"], errors="coerce").max())
            input_manifest["fl_note"] = (
                "Repeated-run summary input: mean and confidence intervals are read from the wrapper's "
                "seed-level aggregation. IDS record-level metrics and FTIL client-round metrics are plotted separately."
            )
        else:
            input_manifest["fl_seed_count"] = int(fl_df["seed"].dropna().nunique())
            input_manifest["fl_note"] = (
                "IDS record-level metrics (e.g., macro-F1) and FTIL client-round metrics are plotted separately. "
                "With only one observed seed, no across-seed confidence interval can be estimated by this script."
            )
    else:
        input_manifest["fl_note"] = "No FL JSON results were loaded; no FL values or figures were invented."

    input_manifest["astb_summary_loaded"] = astb_loaded
    input_manifest["outputs"] = generated
    input_manifest["plotting_environment"] = {
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": getattr(scipy, "__version__", None),
    }
    manifest_path = out / "figure_manifest.json"
    manifest_path.write_text(json.dumps(input_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Figures generated: {len(generated)}")
    for filename in generated:
        print(f"  {out / filename}")
    print(f"Figure manifest: {manifest_path}")
    if provenance_meta.get("available") is False:
        print(f"Provenance figure skipped: {provenance_meta.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
