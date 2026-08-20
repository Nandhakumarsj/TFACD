"""Analyst ground-truth labels for past trust decisions.

Not to be confused with feedback_loop.py, which grid-searches FTIL's
(reject_below_trust, ema_alpha) against the FL-side labeled attack benchmark -
a different subsystem entirely, and one that already has real ground truth
(integrity/benchmark.py's labeled attack scenarios). feedback_loop.py's own
docstring explains why the agentic-side Adaptive Semantic Trust Boundary has
never had an equivalent: no ground truth exists anywhere for "this trust
decision was actually wrong". This module is that missing ground truth
mechanism - a human-labeled record, referenced by AuditEntry.sequence (the
only stable identifier an audit entry already has), so trust_level_thresholds
can eventually be validated against real outcomes (see threshold_validation.py).

Deliberately not authenticated: no login/user-account system exists anywhere
in this codebase (no Flask/FastAPI/user model). analyst_id is free text,
matching this project's current single-workstation trust model - "anyone can
label" is the point, not an oversight.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from tfacd.integrity.certification import chain_hash

DEFAULT_LABELS_PATH = "artifacts/trust_boundary/analyst_labels.jsonl"
_GENESIS_HASH = "0" * 64

LabelValue = Literal["correct", "false_positive", "false_negative", "wrong_trust_level"]


class AnalystLabel(BaseModel):
    label_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    audit_sequence: int
    label: LabelValue
    analyst_id: str
    rationale: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _canonical_bytes(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, default=str).encode("utf-8")


class AnalystLabelStore:
    """Append-only, hash-chained store for human labels - reuses chain_hash()
    from integrity/certification.py (the same primitive AuditLogger uses) so a
    tampered label is as detectable as a tampered audit entry, without a
    second crypto scheme. Not blockchain or truly immutable, same caveat as
    AuditLogger's own docstring.
    """

    def __init__(self, path: str | Path = DEFAULT_LABELS_PATH):
        self.path = Path(path)
        self.last_hash = _GENESIS_HASH
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.last_hash = json.loads(line)["entry_hash"]

    def append(self, label: AnalystLabel) -> str:
        label_dict = label.model_dump(mode="json")
        entry_hash = chain_hash(self.last_hash, _canonical_bytes(label_dict))
        record = {"label": label_dict, "entry_hash": entry_hash, "previous_hash": self.last_hash}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        self.last_hash = entry_hash
        return entry_hash

    def load_all(self) -> list[AnalystLabel]:
        if not self.path.exists():
            return []
        labels = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                labels.append(AnalystLabel.model_validate(json.loads(line)["label"]))
        return labels

    def for_sequence(self, audit_sequence: int) -> list[AnalystLabel]:
        return [label for label in self.load_all() if label.audit_sequence == audit_sequence]


def verify_label_chain(path: str | Path = DEFAULT_LABELS_PATH) -> tuple[bool, int | None]:
    """Recomputes the hash chain. Returns (ok, zero-based index of the first
    broken record) - an index rather than an audit_sequence, since labels are
    not required to be recorded in audit_sequence order."""
    p = Path(path)
    if not p.exists():
        return True, None
    previous_hash = _GENESIS_HASH
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        expected = chain_hash(previous_hash, _canonical_bytes(record["label"]))
        if expected != record["entry_hash"] or record["previous_hash"] != previous_hash:
            return False, i
        previous_hash = record["entry_hash"]
    return True, None
