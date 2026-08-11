from tfacd.runtime.contracts import TrustDecision
from tfacd.trust_boundary.output_protection import find_sensitive_spans, redact, sanitize_decision


def test_ip_addresses_are_not_flagged():
    text = "Blocking source 192.168.0.101 targeting plc-01, destination 10.0.0.5."
    assert find_sensitive_spans(text) == []
    assert redact(text) == text


def test_aws_key_and_password_are_flagged_and_redacted():
    text = "leaked AKIAABCDEFGHIJKLMNOP and password=hunter2 in the rationale"
    findings = find_sensitive_spans(text)
    labels = {label for label, _ in findings}
    assert "aws_access_key" in labels
    assert "password" in labels
    redacted = redact(text)
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "hunter2" not in redacted


def test_email_is_flagged():
    findings = find_sensitive_spans("contact soc-oncall@example.com for review")
    assert any(label == "email" for label, _ in findings)


def test_sanitize_decision_redacts_rationale():
    decision = TrustDecision(
        incident_id="i", accepted=True, terminal_stage="output_protection",
        rationale="password=hunter2 was used to authenticate",
    )
    sanitized = sanitize_decision(decision)
    assert "hunter2" not in sanitized.rationale
    assert sanitized.incident_id == decision.incident_id
