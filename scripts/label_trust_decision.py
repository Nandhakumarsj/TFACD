"""CLI entry point for recording an analyst's ground-truth label on a past
trust decision (analytics/trust_labels.py) - the "anyone can label" mechanism
this project's README calls out as missing. Prints the AuditEntry being
labeled before committing, so the analyst can see exactly what they're
labeling. No login system exists anywhere in this repo, so --analyst-id is
free text - "anyone can label" is the point, not an oversight.
"""

from __future__ import annotations

import argparse
import sys

from tfacd.analytics.kpi import DEFAULT_AUDIT_LOG, load_entries
from tfacd.analytics.trust_labels import DEFAULT_LABELS_PATH, AnalystLabel, AnalystLabelStore

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG)
parser.add_argument("--labels-path", default=DEFAULT_LABELS_PATH)
parser.add_argument("--sequence", type=int, required=True, help="AuditEntry.sequence to label")
parser.add_argument("--label", required=True, choices=["correct", "false_positive", "false_negative", "wrong_trust_level"])
parser.add_argument("--analyst-id", required=True)
parser.add_argument("--rationale", required=True)
args = parser.parse_args()

entries_by_seq = {entry.sequence: entry for entry in load_entries(args.audit_log)}
entry = entries_by_seq.get(args.sequence)
if entry is None:
    print(f"No AuditEntry with sequence={args.sequence} found in {args.audit_log}", file=sys.stderr)
    sys.exit(1)

decision = entry.decision
print(f"Labeling AuditEntry sequence={entry.sequence} (incident_id={entry.incident_id}, agent_id={entry.agent_id})")
print(f"  terminal_stage={decision.terminal_stage} accepted={decision.accepted} trust_level={decision.trust_level}")
print(f"  autonomy_mode={decision.autonomy_mode} executor_mode={decision.executor_mode} engine={decision.engine}")
print(f"  executed_actions={decision.executed_actions}")
print(f"  rationale={decision.rationale!r}")

label = AnalystLabel(
    audit_sequence=args.sequence,
    label=args.label,
    analyst_id=args.analyst_id,
    rationale=args.rationale,
)
entry_hash = AnalystLabelStore(args.labels_path).append(label)

print(f"\nRecorded label {label.label_id} ({args.label!r} by {args.analyst_id!r}) -> {args.labels_path} (entry_hash={entry_hash[:12]}...)")
