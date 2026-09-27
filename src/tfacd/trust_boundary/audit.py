from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tfacd.integrity.certification import chain_hash
from tfacd.runtime.contracts import AuditEntry, TrustDecision

_GENESIS_HASH = "0" * 64


def _canonical_bytes(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, default=str).encode("utf-8")


def _canonical_timestamp(value: datetime | str) -> str:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class AuditLogger:
    """Hash-chained append-only audit trail - tamper-evident, not blockchain or
    truly immutable. Each entry embeds the full TrustDecision, which is the
    "Policy Trace" (diagram 1 lists Trust History a second time under
    Continuous Audit & Compliance: a shared artifact, not a separate engine).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.sequence = 0
        self.last_hash = _GENESIS_HASH
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.sequence = entry["sequence"]
                    self.last_hash = entry["entry_hash"]

    def append(self, decision: TrustDecision, agent_id: str | None = None) -> AuditEntry:
        self.sequence += 1
        timestamp = datetime.now(timezone.utc)
        provenance = {
            "sequence": self.sequence,
            "timestamp": _canonical_timestamp(timestamp),
            "incident_id": decision.incident_id,
            "agent_id": agent_id,
        }
        decision_dict = decision.model_dump(mode="json")
        payload = {"provenance": provenance, "decision": decision_dict}
        entry_hash = chain_hash(self.last_hash, _canonical_bytes(payload))
        entry = AuditEntry(
            sequence=self.sequence,
            timestamp=timestamp,
            incident_id=decision.incident_id,
            agent_id=agent_id,
            entry_hash=entry_hash,
            previous_hash=self.last_hash,
            decision=decision,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        self.last_hash = entry_hash
        return entry


def verify_chain(path: str | Path) -> tuple[bool, int | None]:
    """Recomputes the hash chain over a log file. Returns (ok, first_broken_sequence)."""
    previous_hash = _GENESIS_HASH
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        provenance = {
            "sequence": entry["sequence"],
            "timestamp": _canonical_timestamp(entry["timestamp"]),
            "incident_id": entry["incident_id"],
            "agent_id": entry.get("agent_id"),
        }
        payload = {"provenance": provenance, "decision": entry["decision"]}
        expected = chain_hash(previous_hash, _canonical_bytes(payload))
        if expected != entry["entry_hash"] or entry["previous_hash"] != previous_hash:
            return False, entry["sequence"]
        previous_hash = entry["entry_hash"]
    return True, None
