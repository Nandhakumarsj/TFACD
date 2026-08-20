"""Composes multiple LivePacedSource replays (a normal-traffic baseline, an
attack file injected mid-stream, optionally back to baseline) into one
ordered, time-paced RecordSource - the substrate for
scripts/run_attack_scenario.py's "network disconnect / IP isolation"
simulation. Each ScenarioStep is independently paced from its own paired
PCAP; the whole thing satisfies streaming.sources.RecordSource, so
StreamingIDS.run() consumes it exactly like any other source, unmodified.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tfacd.streaming.live_source import LivePacedSource


@dataclass
class ScenarioStep:
    label: str
    csv_path: str
    pcap_path: str
    max_records: int | None = None
    speed_multiplier: float = 60.0


@dataclass
class Scenario:
    name: str
    description: str
    steps: list[ScenarioStep]
    # Informational only - reported, never forced. None for a diagnostic
    # scenario with no attack-response capability to check for (e.g.
    # modbus_normal_traffic_generalization_check in attack_scenarios.yaml).
    expected_capability: str | None


class ScenarioRunner:
    """RecordSource-conforming (records() -> Iterator[dict])."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def records(self) -> Iterator[dict]:
        for step in self.scenario.steps:
            source = LivePacedSource(
                step.csv_path, step.pcap_path, speed_multiplier=step.speed_multiplier, max_records=step.max_records,
            )
            yield from source.records()


def load_scenarios(path: str | Path) -> dict[str, Scenario]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenarios = {}
    for name, entry in raw.items():
        steps = [ScenarioStep(**step) for step in entry["steps"]]
        scenarios[name] = Scenario(name=name, description=entry["description"], steps=steps, expected_capability=entry["expected_capability"])
    return scenarios
