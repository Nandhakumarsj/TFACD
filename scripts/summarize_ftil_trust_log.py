"""Human-readable summary of the FTIL trust log (artifacts/models/ftil_trust_log.jsonl),
written every round by IntegrityAwareStrategy._log_trust_evidence but never
previously read back by anything - this closes that gap.

Surfaces the three distinct per-client FL signals side by side, since they answer
different questions (see integrity/detector.py's PCAClusterEMAFilter docstring):
- trust_scores: EMA-smoothed personalized client trust (0..1), carries across rounds.
- ood_scores: this round's distance from the cohort's robust center in PCA space,
  as a multiple of the cohort's median distance - memoryless, continuous.
- detection: which clustering technique produced both, plus its own quality
  signals (explained variance, silhouette).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--trust-log", default="artifacts/models/ftil_trust_log.jsonl")
args = parser.parse_args()

log_path = Path(args.trust_log)
if not log_path.exists():
    raise SystemExit(f"No trust log at {log_path} - run a federated training round with use-ftil=true first.")

rounds = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rounds:
    raise SystemExit(f"{log_path} exists but is empty.")

print(f"{len(rounds)} round(s) logged at {log_path}\n")

print(f"{'round':>5} {'cluster_method':<20} {'n':>3} {'expl_var':>9} {'silhouette':>10} {'accepted':>8} {'rej_val':>7} {'rej_det':>7} {'max_ood':>8}")
for entry in rounds:
    detection = entry.get("detection") or {}
    ood_scores = entry.get("ood_scores") or {}
    max_ood = max(ood_scores.values()) if ood_scores else 0.0
    silhouette = detection.get("silhouette")
    print(
        f"{entry['round']:>5} {detection.get('cluster_method', 'n/a'):<20} {detection.get('n_clients', 0):>3} "
        f"{detection.get('explained_variance_ratio', 0.0):>9.3f} "
        f"{'n/a' if silhouette is None else f'{silhouette:.3f}':>10} "
        f"{len(entry.get('accepted', [])):>8} {len(entry.get('rejected_validation', [])):>7} "
        f"{len(entry.get('rejected_detector', [])):>7} {max_ood:>8.2f}"
    )

# Per-client personalization trajectory: how each client's EMA trust score
# evolved round over round - the one signal that carries history.
trajectories: dict[str, dict[int, float]] = defaultdict(dict)
for entry in rounds:
    for client_id, score in (entry.get("trust_scores") or {}).items():
        trajectories[client_id][entry["round"]] = score

print(f"\nper-client trust trajectory ({len(trajectories)} client(s) seen):")
for client_id in sorted(trajectories, key=lambda c: (len(c), c)):
    series = trajectories[client_id]
    ordered = [f"{series[r]:.2f}" for r in sorted(series)]
    flagged = " <- currently below reject threshold" if ordered and float(ordered[-1]) < 0.35 else ""
    print(f"  {client_id:<12} {' -> '.join(ordered)}{flagged}")

flagged_clients = [
    client_id for client_id, score in rounds[-1].get("trust_scores", {}).items() if score < 0.35
]
if flagged_clients:
    print(f"\n{len(flagged_clients)} client(s) below the 0.35 reject threshold as of the last round: {flagged_clients}")
