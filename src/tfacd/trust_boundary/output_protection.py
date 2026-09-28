from __future__ import annotations

import re

from tfacd.runtime.contracts import TrustDecision

# IPv4 addresses are deliberately NOT matched here - target/source_id fields
# legitimately contain them in this domain, they're not leaked secrets.
_PATTERNS = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "api_key": re.compile(r"\bapi[_-]?key\s*[:=]\s*\S+", re.IGNORECASE),
    "password": re.compile(r"\bpassword\s*[:=]\s*\S+", re.IGNORECASE),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def find_sensitive_spans(text: str) -> list[tuple[str, str]]:
    """Returns (label, matched_text) for every sensitive-looking span found."""
    return [(label, match.group(0)) for label, pattern in _PATTERNS.items() for match in pattern.finditer(text)]


def redact(text: str) -> str:
    redacted = text
    for pattern in _PATTERNS.values():
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def sanitize_decision(decision: TrustDecision) -> TrustDecision:
    """Sanitizes the outgoing TrustDecision unconditionally as the last step
    before it's returned/logged - even a blocked decision's echoed rationale
    could contain something worth redacting.

    REVIEW FIX (Section 26 - leak path): this used to sanitize ONLY
    `rationale`. preprocessing.py's obfuscation detector can embed a decoded
    preview of a flagged value directly into a StageResult's `reasons` (e.g.
    "looks base64-encoded (decodes to: ...)") - if that decoded content
    happens to contain something matching one of the patterns above (an API
    key, a password, an email), it would reach the audit log completely
    unredacted through `reasons` while `rationale` was the only field actually
    protected. Every free-text field that can carry attacker- or
    detector-echoed content must get the same treatment before this decision
    is returned or audited.
    """
    sanitized_stage_results = [
        stage_result.model_copy(update={"reasons": [redact(reason) for reason in stage_result.reasons]})
        for stage_result in decision.stage_results
    ]
    return decision.model_copy(
        update={"rationale": redact(decision.rationale), "stage_results": sanitized_stage_results}
    )
