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
    """
    return decision.model_copy(update={"rationale": redact(decision.rationale)})
