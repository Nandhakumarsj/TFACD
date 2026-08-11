from __future__ import annotations

from pathlib import Path

import yaml

from tfacd.runtime.contracts import IDSAlert, ThreatContext


class ThreatContextGenerator:
    def __init__(self, mapping_path: str | Path):
        self.mapping = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8"))

    def enrich(self, alert: IDSAlert) -> ThreatContext:
        entry = self.mapping.get(alert.attack_type) or self.mapping.get("Normal")
        return ThreatContext(alert=alert, **entry)
