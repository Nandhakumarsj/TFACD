from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="Create individual second-review figures from machine-readable experiment outputs.")
parser.add_argument("--ablation", default="artifacts/second_review/astb/component_ablation_summary.json")
parser.add_argument("--fl", default="artifacts/second_review/fl_security_matrix.json")
parser.add_argument("--output-dir", default="artifacts/second_review_figures")
args = parser.parse_args()

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)

if Path(args.ablation).exists():
    summary = json.loads(Path(args.ablation).read_text(encoding="utf-8"))
    names = list(summary["variants"])
    far = [summary["variants"][n]["false_allow_rate"] * 100 for n in names]
    uabr = [summary["variants"][n]["unsafe_action_blocking_rate"] * 100 for n in names]
    sapr = [summary["variants"][n]["safe_action_preservation_rate"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(names))
    ax.bar(x, far, width=0.25, label="False authorization rate")
    ax.bar([i + 0.25 for i in x], uabr, width=0.25, label="Unsafe action blocking rate")
    ax.bar([i + 0.50 for i in x], sapr, width=0.25, label="Safe action preservation rate")
    ax.set_xticks([i + 0.25 for i in x], names, rotation=20, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("ASTB component ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "astb_component_ablation_rates.png", dpi=220)
    plt.close(fig)

if Path(args.fl).exists():
    data = json.loads(Path(args.fl).read_text(encoding="utf-8"))
    df = pd.DataFrame([
        {
            "mode": r["mode"],
            "attack": r["attack"],
            "fraction": r["malicious_fraction_requested"] * 100,
            "macro_f1": r["metrics"]["macro_f1"] * 100,
        }
        for r in data["results"]
    ])
    for (mode, attack), group in df.groupby(["mode", "attack"]):
        group = group.sort_values("fraction")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(group["fraction"], group["macro_f1"], marker="o", label=f"{mode} / {attack}")
        ax.set_xlabel("Malicious clients (%)")
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_title(f"FL robustness: {mode} under {attack}")
        ax.set_xticks([0, 10, 20, 30])
        ax.legend()
        fig.tight_layout()
        safe = f"fl_{mode}_{attack}".replace("/", "_")
        fig.savefig(out / f"{safe}.png", dpi=220)
        plt.close(fig)

print(f"Figures written to {out.resolve()}")
