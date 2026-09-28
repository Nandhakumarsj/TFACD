import json

import pytest

from tfacd.runtime.contracts import TrustDecision
from tfacd.trust_boundary.audit import AuditChainTamperedError, AuditLogger, verify_chain


def make_decision(incident_id):
    return TrustDecision(incident_id=incident_id, accepted=True, terminal_stage="capability_enforcement", rationale="ok")


def test_chain_verifies_when_untampered(tmp_path):
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path)
    for i in range(5):
        logger.append(make_decision(f"inc-{i}"))

    ok, bad_sequence = verify_chain(path)
    assert ok
    assert bad_sequence is None


def test_tamper_is_detected_from_tampered_entry_onward(tmp_path):
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path)
    for i in range(5):
        logger.append(make_decision(f"inc-{i}"))

    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[2])  # sequence 3
    tampered["decision"]["accepted"] = not tampered["decision"]["accepted"]
    lines[2] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad_sequence = verify_chain(path)
    assert not ok
    assert bad_sequence == 3


def test_logger_resumes_chain_across_instances(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLogger(path).append(make_decision("inc-0"))

    second = AuditLogger(path)
    entry = second.append(make_decision("inc-1"))
    assert entry.sequence == 2
    assert entry.previous_hash != "0" * 64

    ok, _ = verify_chain(path)
    assert ok


def test_tamper_in_provenance_fields_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path)
    logger.append(make_decision("inc-0"), agent_id="agent-a")

    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["agent_id"] = "agent-b"
    lines[0] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad_sequence = verify_chain(path)
    assert not ok
    assert bad_sequence == 1


def test_new_logger_refuses_to_append_onto_tampered_existing_log(tmp_path):
    """REVIEW FIX (P1 - startup trusts the log blindly): AuditLogger used to
    read only the last stored hash on startup and keep extending the chain
    from it without ever checking the ON-DISK chain was internally
    consistent. A log tampered with before this process started must now be
    detected at construction time, and append() must refuse to extend it."""
    path = tmp_path / "audit.jsonl"
    AuditLogger(path).append(make_decision("inc-0"))

    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["decision"]["accepted"] = not tampered["decision"]["accepted"]
    lines[0] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reopened = AuditLogger(path)
    assert reopened._tampered is True
    with pytest.raises(AuditChainTamperedError):
        reopened.append(make_decision("inc-1"))
