from __future__ import annotations

from pathlib import Path
from typing import Any

from tfacd.integrity.certification import verify_manifest, write_manifest
from tfacd.trust_boundary.output_protection import redact


def certify_history_snapshot(history_path: str | Path, manifest_path: str | Path | None = None) -> Path:
    """Periodic whole-file provenance check, not per-request - catches wholesale
    file replacement that audit.py's per-entry hash chain alone wouldn't (a
    replaced file with an internally-consistent chain still validates).
    """
    return write_manifest(history_path, {"purpose": "trust_boundary_history_provenance"}, manifest_path)


def verify_history_provenance(history_path: str | Path, manifest_path: str | Path) -> bool:
    return verify_manifest(history_path, manifest_path)


def sanitize_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Guards every write into EntityHistory - reuses output_protection's
    detector rather than reimplementing sensitive-data scanning."""
    return {key: (redact(value) if isinstance(value, str) else value) for key, value in payload.items()}


def detect_implausible_entries(events: list[dict], max_trust_jump: float = 0.9) -> list[str]:
    """Heuristic, not proof: flags an implausibly large trust-value jump between
    consecutive trust_decision events for the same entity, worth a review.
    """
    warnings: list[str] = []
    previous_trust: float | None = None
    for event in events:
        if event["kind"] != "trust_decision":
            continue
        trust_value = event["payload"].get("trust_value")
        if trust_value is None:
            continue
        if previous_trust is not None and abs(trust_value - previous_trust) > max_trust_jump:
            warnings.append(f"implausible trust jump at {event['timestamp']}: {previous_trust} -> {trust_value}")
        previous_trust = trust_value
    return warnings
