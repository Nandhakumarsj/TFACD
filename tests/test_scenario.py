import struct

import pytest
import yaml

from tfacd.streaming.scenario import Scenario, ScenarioRunner, ScenarioStep, load_scenarios


def _write_pcap(path, timestamps):
    with path.open("wb") as handle:
        handle.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for ts in timestamps:
            sec, usec = int(ts), round((ts - int(ts)) * 1_000_000)
            handle.write(struct.pack("<IIII", sec, usec, 0, 0))


def _write_csv(path, values):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("val\n")
        for v in values:
            handle.write(f"{v}\n")


def _write_pair(tmp_path, name, values, base_ts=1_700_000_000.0):
    csv_path, pcap_path = tmp_path / f"{name}.csv", tmp_path / f"{name}.pcap"
    _write_csv(csv_path, values)
    _write_pcap(pcap_path, [base_ts + i * 0.01 for i in range(len(values))])
    return csv_path, pcap_path


def test_scenario_runner_concatenates_steps_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr("tfacd.streaming.live_source.time.sleep", lambda s: None)
    baseline_csv, baseline_pcap = _write_pair(tmp_path, "baseline", ["n1", "n2"])
    attack_csv, attack_pcap = _write_pair(tmp_path, "attack", ["a1", "a2", "a3"])

    scenario = Scenario(
        name="test",
        description="baseline then attack",
        expected_capability="block_source",
        steps=[
            ScenarioStep(label="baseline", csv_path=str(baseline_csv), pcap_path=str(baseline_pcap), speed_multiplier=1000.0),
            ScenarioStep(label="attack", csv_path=str(attack_csv), pcap_path=str(attack_pcap), speed_multiplier=1000.0),
        ],
    )

    records = list(ScenarioRunner(scenario).records())

    assert [r["val"] for r in records] == ["n1", "n2", "a1", "a2", "a3"]


def test_scenario_step_max_records_is_respected(tmp_path, monkeypatch):
    monkeypatch.setattr("tfacd.streaming.live_source.time.sleep", lambda s: None)
    csv_path, pcap_path = _write_pair(tmp_path, "data", ["r1", "r2", "r3", "r4", "r5"])

    scenario = Scenario(
        name="test", description="d", expected_capability="block_source",
        steps=[ScenarioStep(label="s", csv_path=str(csv_path), pcap_path=str(pcap_path), max_records=2, speed_multiplier=1000.0)],
    )

    records = list(ScenarioRunner(scenario).records())

    assert [r["val"] for r in records] == ["r1", "r2"]


def test_load_scenarios_parses_real_yaml_shape(tmp_path):
    scenarios_path = tmp_path / "scenarios.yaml"
    scenarios_path.write_text(
        yaml.safe_dump(
            {
                "my_scenario": {
                    "description": "a test scenario",
                    "expected_capability": "isolate_segment",
                    "steps": [
                        {"label": "baseline", "csv_path": "a.csv", "pcap_path": "a.pcap", "max_records": 10, "speed_multiplier": 30.0},
                        {"label": "attack", "csv_path": "b.csv", "pcap_path": "b.pcap"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    scenarios = load_scenarios(scenarios_path)

    assert set(scenarios) == {"my_scenario"}
    scenario = scenarios["my_scenario"]
    assert scenario.expected_capability == "isolate_segment"
    assert len(scenario.steps) == 2
    assert scenario.steps[0].max_records == 10
    assert scenario.steps[1].max_records is None  # default, not required in the YAML


def test_real_attack_scenarios_config_loads_and_references_real_dataset_paths():
    """configs/attack_scenarios.yaml itself must parse and every referenced
    csv_path/pcap_path must be a real path under datasets/Edge_IIoTset (skips
    the actual-file-exists check if the (large, gitignored) dataset isn't
    present on this machine - this test only guards the YAML/path shape)."""
    scenarios = load_scenarios("configs/attack_scenarios.yaml")
    assert len(scenarios) >= 2
    for scenario in scenarios.values():
        assert len(scenario.steps) >= 1
        for step in scenario.steps:
            assert "datasets/Edge_IIoTset" in step.csv_path
            assert "datasets/Edge_IIoTset" in step.pcap_path
