"""Narrow grid search over FTIL's PCAClusterEMAFilter parameters against Gate
4's labeled attack benchmark. See analytics/feedback_loop.py's module
docstring for the two scope caveats this script's output must be read with:
(1) tunes FTIL's detector thresholds, not the agentic Trust Boundary's Rs/Rc/Rb/T
thresholds - no ground truth exists for the latter; (2) "after" numbers below
measure the isolated pca_cluster_ema detector (weighted_average aggregation),
not the live IntegrityAwareStrategy (trimmed_mean aggregation).
"""

from __future__ import annotations

import argparse
import time

from tfacd.analytics.feedback_loop import BASELINE_FPR, BASELINE_TPR, select_best_candidate, run_grid_search
from tfacd.common.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--fpr-tolerance", type=float, default=0.1)
args = parser.parse_args()

config = load_config(args.config)


def report_progress(result) -> None:
    m = result.detection_metrics
    print(
        f"[{time.strftime('%H:%M:%S')}] reject_below_trust={result.params['reject_below_trust']:.2f} "
        f"ema_alpha={result.params['ema_alpha']:.2f}  ->  tpr={m['tpr']} fpr={m['fpr']}"
    )


print(f"Running {3 * 3} candidates (2 rounds x sign_flip/gaussian_noise x pca_cluster_ema each) - this takes real GPU/CPU training time, expect several minutes.")
results = run_grid_search(config, on_progress=report_progress)
chosen, met_constraint = select_best_candidate(results, fpr_tolerance=args.fpr_tolerance)

print()
print("ISOLATED-DETECTOR caveat: all numbers below are pca_cluster_ema's own TPR/FPR under")
print("weighted_average aggregation (integrity/benchmark.py). They do NOT measure the live")
print("federated/integrity_strategy.py pipeline, which uses trimmed_mean aggregation instead")
print("specifically because Gate 4 found trimmed_mean alone already holds ~0.93-0.94 macro-F1")
print("under every attack tested, regardless of detector threshold. A TPR/FPR gain here is a")
print("claim about the isolated detector, not about live end-to-end robustness.")
print()
print(f"{'':<6}{'reject_below_trust':>20}{'ema_alpha':>12}{'tpr':>10}{'fpr':>10}")
print(f"{'BEFORE':<6}{'0.35 (default)':>20}{'0.30 (default)':>12}{BASELINE_TPR:>10.3f}{BASELINE_FPR:>10.3f}")
after_tpr = chosen.detection_metrics["tpr"]
after_fpr = chosen.detection_metrics["fpr"]
print(f"{'AFTER':<6}{chosen.params['reject_below_trust']:>20.2f}{chosen.params['ema_alpha']:>12.2f}{after_tpr:>10.3f}{after_fpr:>10.3f}")
print()
if met_constraint:
    print(f"Selected candidate meets the fpr <= {args.fpr_tolerance} constraint and maximizes tpr among those that do.")
else:
    print(f"No candidate met fpr <= {args.fpr_tolerance}; reporting the best tpr-fpr tradeoff found instead.")
