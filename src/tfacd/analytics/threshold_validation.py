"""Validates trust_level_thresholds against real analyst-labeled outcomes.

This is an operator-run diagnostic that REPORTS, it does not auto-tune
trust_level_thresholds - changing a live safety threshold stays a human
decision, same posture as BehavioralTrustEngine.refit_from_history() (which
also gates on a minimum real-sample count before doing anything, and is only
ever invoked explicitly, never from the live pipeline).

Before enough labels exist, this intentionally reports "not enough labels
yet" rather than a misleadingly precise-looking number from a handful of
samples - the exact overclaiming this project's research-honesty conventions
warn against everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tfacd.analytics.kpi import DEFAULT_AUDIT_LOG, load_entries
from tfacd.analytics.trust_labels import DEFAULT_LABELS_PATH, AnalystLabel, AnalystLabelStore
from tfacd.runtime.contracts import AuditEntry

MIN_SAMPLES = 20
_TRUST_LEVELS = ("low", "medium", "high", "verified")


@dataclass
class TrustLevelBreakdown:
    trust_level: str
    num_labels: int
    num_correct: int
    agreement_rate: float


@dataclass
class ProductionActionReview:
    """A labeled decision whose executed_actions came from a real (not
    simulated) executor - the highest-priority row in this report, since it
    already had a real-world effect, not just a logged recommendation."""

    audit_sequence: int
    label: str
    analyst_id: str
    executed_actions: list[str]


@dataclass
class ThresholdValidationReport:
    num_labels: int
    num_labeled_decisions: int  # distinct audit_sequence values with >=1 label
    ready: bool  # num_labeled_decisions >= MIN_SAMPLES
    label_counts: dict[str, int]
    per_trust_level: list[TrustLevelBreakdown]  # only meaningful when ready
    production_action_reviews: list[ProductionActionReview]  # always surfaced, regardless of `ready`
    unmatched_labels: list[int]  # audit_sequence values with a label but no matching audit entry


def _index_by_sequence(entries: list[AuditEntry]) -> dict[int, AuditEntry]:
    return {entry.sequence: entry for entry in entries}


def compute_report(
    audit_log_path: str | Path = DEFAULT_AUDIT_LOG,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    min_samples: int = MIN_SAMPLES,
) -> ThresholdValidationReport:
    entries_by_seq = _index_by_sequence(load_entries(audit_log_path))
    labels = AnalystLabelStore(labels_path).load_all()

    label_counts: dict[str, int] = {}
    unmatched_labels: list[int] = []
    production_reviews: list[ProductionActionReview] = []
    by_level: dict[str, list[AnalystLabel]] = {level: [] for level in _TRUST_LEVELS}

    labeled_sequences: set[int] = set()
    for label in labels:
        label_counts[label.label] = label_counts.get(label.label, 0) + 1
        entry = entries_by_seq.get(label.audit_sequence)
        if entry is None:
            unmatched_labels.append(label.audit_sequence)
            continue
        labeled_sequences.add(label.audit_sequence)

        if entry.decision.executor_mode == "production":
            production_reviews.append(
                ProductionActionReview(
                    audit_sequence=label.audit_sequence,
                    label=label.label,
                    analyst_id=label.analyst_id,
                    executed_actions=entry.decision.executed_actions,
                )
            )

        if entry.decision.trust_level in by_level:
            by_level[entry.decision.trust_level].append(label)

    ready = len(labeled_sequences) >= min_samples
    per_trust_level = []
    if ready:
        for level in _TRUST_LEVELS:
            level_labels = by_level[level]
            if not level_labels:
                continue
            num_correct = sum(1 for label in level_labels if label.label == "correct")
            per_trust_level.append(
                TrustLevelBreakdown(
                    trust_level=level,
                    num_labels=len(level_labels),
                    num_correct=num_correct,
                    agreement_rate=num_correct / len(level_labels),
                )
            )

    return ThresholdValidationReport(
        num_labels=len(labels),
        num_labeled_decisions=len(labeled_sequences),
        ready=ready,
        label_counts=label_counts,
        per_trust_level=per_trust_level,
        production_action_reviews=production_reviews,
        unmatched_labels=unmatched_labels,
    )
