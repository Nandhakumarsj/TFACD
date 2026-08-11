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


def _canonicalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")  # strip zero-width/format chars


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
    return None


def run(plan: CyberActionPlan, session: SessionContext, history: EntityHistory, config: dict[str, Any]) -> tuple[StageResult, CyberActionPlan]:
    reasons: list[str] = []

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
                value = _canonicalize(value)
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

    normalized_plan = plan.model_copy(update={"actions": normalized_actions, "rationale": _canonicalize(plan.rationale)})
    return StageResult(stage="preprocessing", accepted=not reasons, reasons=reasons), normalized_plan
