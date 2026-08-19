from __future__ import annotations

import base64
import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any

from tfacd.agentic.history import EntityHistory
from tfacd.runtime.contracts import CyberActionPlan, SessionContext, StageResult

_BASE64_RUN = re.compile(r"^[A-Za-z0-9+/]{16,}={0,2}$")
_HEX_RUN = re.compile(r"^(?:[0-9a-fA-F]{2}){8,}$")
_URL_ENCODED = re.compile(r"%[0-9a-fA-F]{2}")

# Deliberately narrow: only flags a SUBSTITUTED dangerous keyword (e.g.
# "1gn0r3" -> "ignore"), never "any digit near a letter" - a broader heuristic
# would false-positive on ordinary alphanumeric content like device names
# ("gateway-01", "vlan10"). Vocabulary is the kind of instruction-override
# language a hidden prompt-injection payload would use, not a general profanity
# filter.
_LEETSPEAK_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "@": "a", "!": "i"})
_DANGEROUS_KEYWORDS = (
    "ignore", "override", "bypass", "disregard", "admin", "root", "shutdown",
    "delete", "execute", "system", "disable", "unlock", "sudo",
)
_KEYWORD_PATTERNS = {word: re.compile(rf"\b{re.escape(word)}\b") for word in _DANGEROUS_KEYWORDS}


def canonicalize(text: str) -> str:
    """Unicode NFKC normalization + zero-width/format-character stripping.

    Public (not `_`-prefixed) because it has a second caller outside this
    module: streaming/pipeline.py applies it to IDSAlert.source_id/target_asset
    at alert-construction time - the one place those fields, sourced from raw
    ip.src_host/ip.dst_host record values, get sanitized before they reach
    every downstream consumer (the LLM decision engine's prompt, semantic_risk
    scoring, decision_engine's rationale templates) in one shot, rather than
    patching each consumer separately and risking missing one.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")  # strip zero-width/format chars


def _detect_leetspeak(text: str) -> str | None:
    """Flags a leetspeak-substituted dangerous keyword that was NOT already
    plainly present in the raw text - i.e. was hidden by the substitution, not
    a word that would already read as suspicious unsubstituted."""
    lowered = text.lower()
    de_leeted = lowered.translate(_LEETSPEAK_MAP)
    if de_leeted == lowered:
        return None  # nothing was actually substituted
    for word, pattern in _KEYWORD_PATTERNS.items():
        if pattern.search(de_leeted) and not pattern.search(lowered):
            return f"looks leetspeak-obfuscated (decodes to contain {word!r})"
    return None


def _detect_obfuscation(text: str) -> str | None:
    stripped = text.strip()
    if _BASE64_RUN.match(stripped):
        try:
            decoded = base64.b64decode(stripped, validate=True).decode("utf-8", errors="replace")
            return f"looks base64-encoded (decodes to: {decoded[:60]!r})"
        except (ValueError, UnicodeDecodeError):
            pass
    if _HEX_RUN.match(stripped):
        return "looks hex-encoded"
    if _URL_ENCODED.search(stripped):
        return "contains URL-percent-encoding"
    leetspeak = _detect_leetspeak(stripped)
    if leetspeak:
        return leetspeak
    return None


def run(plan: CyberActionPlan, session: SessionContext, history: EntityHistory, config: dict[str, Any]) -> tuple[StageResult, CyberActionPlan]:
    reasons: list[str] = []

    # Nonce replay: a session is only ever fresh within session_max_age_seconds
    # (below), so a resent (session_id, nonce, issued_at) tuple can only ever be a
    # genuine replay attempt within that same window - nothing older needs to be
    # remembered. Recorded unconditionally (before any other stage-1 check), since
    # the nonce was "spent" the moment it was offered, regardless of whether this
    # particular plan goes on to be accepted.
    replay_window = timedelta(seconds=config["session_max_age_seconds"])
    prior_nonces = history.recent(session.agent_id, kind="nonce_used", within=replay_window)
    if any(e["payload"].get("nonce") == session.nonce for e in prior_nonces):
        reasons.append(f"nonce replay detected: {session.nonce!r} already used within the last {replay_window}")
    else:
        history.append(session.agent_id, "nonce_used", {"nonce": session.nonce, "session_id": session.session_id})

    session_age = datetime.now(timezone.utc) - session.issued_at
    if session_age > timedelta(seconds=config["session_max_age_seconds"]):
        reasons.append(f"session expired: issued {session_age} ago")
    if session_age < timedelta(0):
        reasons.append("session issued_at is in the future")

    if not plan.actions:
        reasons.append("plan has no actions")
    if len(plan.actions) > config["max_actions_per_plan"]:
        reasons.append(f"too many actions: {len(plan.actions)} > {config['max_actions_per_plan']}")

    normalized_actions = []
    for action in plan.actions:
        normalized_params: dict[str, Any] = {}
        for key, value in action.parameters.items():
            if isinstance(value, str):
                if len(value) > config["max_parameter_string_length"]:
                    reasons.append(f"parameter '{key}' exceeds max length")
                    value = value[: config["max_parameter_string_length"]]
                value = canonicalize(value)
                obfuscated = _detect_obfuscation(value)
                if obfuscated:
                    reasons.append(f"parameter '{key}' {obfuscated}")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    reasons.append(f"parameter '{key}' is non-finite")
                elif abs(value) > config["max_numeric_parameter"]:
                    reasons.append(f"parameter '{key}' exceeds max magnitude")
            normalized_params[key] = value
        normalized_actions.append(action.model_copy(update={"parameters": normalized_params}))

    recent_decisions = history.count_since(session.agent_id, within=timedelta(hours=1), kind="trust_decision")
    quota = config["entity_action_quota_per_hour"]
    if recent_decisions >= quota:
        reasons.append(f"hourly action quota exceeded ({recent_decisions} >= {quota})")

    normalized_plan = plan.model_copy(update={"actions": normalized_actions, "rationale": canonicalize(plan.rationale)})
    return StageResult(stage="preprocessing", accepted=not reasons, reasons=reasons), normalized_plan
