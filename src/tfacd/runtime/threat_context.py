from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import yaml

from tfacd.runtime.contracts import IDSAlert, ThreatContext

logger = logging.getLogger(__name__)


class ThreatContextGenerator:
    """Silence here is exactly how an unmapped attack_type used to fall through
    to Normal (informational/observe) without anyone noticing - so this warns
    loudly at both construction time (once, for the whole mapping) and per-alert
    (for the actual fallback), and can be made to raise instead via `strict`.
    """

    def __init__(self, mapping_path: str | Path, *, strict: bool = False, known_classes: Sequence[str] | None = None):
        self.mapping = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8"))
        self.strict = strict
        if known_classes is not None:
            unmapped = sorted(set(known_classes) - set(self.mapping))
            dead = sorted(set(self.mapping) - set(known_classes))
            if unmapped:
                logger.warning("threat_context mapping is missing %d known class(es): %s", len(unmapped), unmapped)
            if dead:
                logger.warning("threat_context mapping has %d key(s) no known class can emit: %s", len(dead), dead)

    def enrich(self, alert: IDSAlert) -> ThreatContext:
        entry = self.mapping.get(alert.attack_type)
        if entry is None:
            if self.strict:
                raise KeyError(f"no threat_context mapping for attack_type={alert.attack_type!r}")
            logger.warning("attack_type=%r not in threat_context mapping, falling back to Normal", alert.attack_type)
            entry = self.mapping["Normal"]
        return ThreatContext(alert=alert, **entry)
