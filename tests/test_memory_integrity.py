from tfacd.trust_boundary.memory_integrity import (
    certify_history_snapshot,
    detect_implausible_entries,
    sanitize_event_payload,
    verify_history_provenance,
)


def test_provenance_roundtrip_and_tamper_detection(tmp_path):
    history_path = tmp_path / "history.jsonl"
    history_path.write_text('{"entity_id": "a"}\n', encoding="utf-8")
    manifest_path = tmp_path / "history.manifest.json"

    certify_history_snapshot(history_path, manifest_path)
    assert verify_history_provenance(history_path, manifest_path)

    history_path.write_text('{"entity_id": "a", "payload": "tampered"}\n', encoding="utf-8")
    assert not verify_history_provenance(history_path, manifest_path)


def test_sanitize_event_payload_redacts_strings_only():
    payload = {"note": "password=hunter2", "count": 3, "accepted": True}
    sanitized = sanitize_event_payload(payload)
    assert "hunter2" not in sanitized["note"]
    assert sanitized["count"] == 3
    assert sanitized["accepted"] is True


def test_detect_implausible_entries_flags_large_jump():
    events = [
        {"kind": "trust_decision", "timestamp": "t1", "payload": {"trust_value": 0.02}},
        {"kind": "trust_decision", "timestamp": "t2", "payload": {"trust_value": 0.98}},
    ]
    assert detect_implausible_entries(events)


def test_detect_implausible_entries_ignores_smooth_changes():
    events = [
        {"kind": "trust_decision", "timestamp": "t1", "payload": {"trust_value": 0.5}},
        {"kind": "trust_decision", "timestamp": "t2", "payload": {"trust_value": 0.55}},
    ]
    assert detect_implausible_entries(events) == []
