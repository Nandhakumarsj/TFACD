from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


class EntityHistory:
    """Per-entity (agent_id) event timeline: incident/action events plus trust-decision outcomes.

    Defaults to in-memory only. An optional JSONL persistence path makes history
    survive across process runs (needed for behavioral trust / policy-violation
    counts to mean anything). Not safe for concurrent multi-process writers, and
    has no retention/compaction yet - fine for a single-process library with no
    HTTP service, revisit if that changes.
    """

    def __init__(self, persist_path: str | Path | None = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self._events: dict[str, list[dict]] = {}
        if self.persist_path and self.persist_path.exists():
            for line in self.persist_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._replay(json.loads(line))

    def _replay(self, event: dict) -> None:
        self._events.setdefault(event["entity_id"], []).append(event)

    def append(self, entity_id: str, kind: str, payload: dict) -> None:
        event = {
            "entity_id": entity_id,
            "kind": kind,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        self._events.setdefault(entity_id, []).append(event)
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self.persist_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

    def recent(self, entity_id: str, kind: str | None = None, within: timedelta | None = None) -> list[dict]:
        events = self._events.get(entity_id, [])
        if kind is not None:
            events = [e for e in events if e["kind"] == kind]
        if within is not None:
            cutoff = datetime.now(timezone.utc) - within
            events = [e for e in events if datetime.fromisoformat(e["timestamp"]) >= cutoff]
        return events

    def count_since(self, entity_id: str, within: timedelta, kind: str | None = None) -> int:
        return len(self.recent(entity_id, kind=kind, within=within))

    def all_events(self, kind: str | None = None) -> list[dict]:
        """Every event across every entity, optionally filtered by kind - used
        by BehavioralTrustEngine.refit_from_history() to reconstruct a real
        observed population without reaching into the private per-entity dict."""
        events = [event for entity_events in self._events.values() for event in entity_events]
        if kind is not None:
            events = [event for event in events if event["kind"] == kind]
        return events
