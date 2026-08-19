"""CLI entry point for validating trust_level_thresholds against real
analyst-labeled outcomes (analytics/threshold_validation.py). Reports only -
never auto-tunes a live safety threshold. See analytics/trust_labels.py and
scripts/label_trust_decision.py for how labels get recorded in the first
place.
"""

from __future__ import annotations

import argparse

from tfacd.analytics.kpi import DEFAULT_AUDIT_LOG
from tfacd.analytics.threshold_validation import MIN_SAMPLES, compute_report
from tfacd.analytics.trust_labels import DEFAULT_LABELS_PATH

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG)
parser.add_argument("--labels-path", default=DEFAULT_LABELS_PATH)
parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
args = parser.parse_args()

report = compute_report(args.audit_log, args.labels_path, min_samples=args.min_samples)

print(f"{report.num_labels} label(s) recorded, covering {report.num_labeled_decisions} distinct trust decision(s)")
print(f"label breakdown: {report.label_counts}")
if report.unmatched_labels:
    print(f"WARNING: {len(report.unmatched_labels)} label(s) reference an audit_sequence not found in {args.audit_log}: {report.unmatched_labels}")

if report.production_action_reviews:
    print(f"\n*** {len(report.production_action_reviews)} labeled decision(s) executed via a REAL (production) executor - review these first: ***")
    for review in report.production_action_reviews:
        print(f"  sequence={review.audit_sequence} label={review.label} analyst={review.analyst_id} executed_actions={review.executed_actions}")

if not report.ready:
    print(f"\nnot enough labels yet: {report.num_labeled_decisions}/{args.min_samples} distinct labeled decisions - trust_level_thresholds remain unvalidated")
else:
    print(f"\ntrust_level_thresholds agreement rate, by level ({report.num_labeled_decisions} labeled decisions):")
    print(f"{'trust_level':<12} {'num_labels':>10} {'num_correct':>11} {'agreement_rate':>15}")
    for row in report.per_trust_level:
        print(f"{row.trust_level:<12} {row.num_labels:>10} {row.num_correct:>11} {row.agreement_rate:>15.3f}")
    print("\nagreement_rate is descriptive, not a tuning signal - trust_level_thresholds (0.40/0.65/0.85) are not changed by this script.")
