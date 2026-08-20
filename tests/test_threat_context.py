import logging

import pytest
import yaml

from tfacd.common.config import load_config
from tfacd.runtime.contracts import IDSAlert
from tfacd.runtime.threat_context import ThreatContextGenerator

MAPPING_PATH = "configs/threat_context.yaml"

# The 15 real classes artifacts/data/metadata.json's "classes" list produces -
# hardcoded rather than read from that gitignored, generated file, so this
# test still runs (and still matters) even before preprocessing has been run.
MODEL_CLASSES = [
    "Backdoor", "DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP", "DDoS_UDP", "Fingerprinting", "MITM", "Normal",
    "Password", "Port_Scanning", "Ransomware", "SQL_injection", "Uploading", "Vulnerability_scanner", "XSS",
]


def test_known_attack_type_maps_severity_and_playbooks():
    generator = ThreatContextGenerator(MAPPING_PATH)
    context = generator.enrich(IDSAlert(attack_type="Port_Scanning", confidence=0.8))
    assert context.severity == "medium"
    assert context.priority == "P2"
    assert "block_source" in context.allowed_playbooks


def test_unknown_attack_type_falls_back_to_normal_and_warns(caplog):
    generator = ThreatContextGenerator(MAPPING_PATH)
    with caplog.at_level(logging.WARNING):
        context = generator.enrich(IDSAlert(attack_type="Totally_Unmapped_Type", confidence=0.5))
    assert context.severity == "informational"
    assert context.allowed_playbooks == ["observe", "log_event"]
    assert any("Totally_Unmapped_Type" in record.message for record in caplog.records)


def test_strict_mode_raises_instead_of_falling_back():
    generator = ThreatContextGenerator(MAPPING_PATH, strict=True)
    with pytest.raises(KeyError):
        generator.enrich(IDSAlert(attack_type="Totally_Unmapped_Type", confidence=0.5))


def test_construction_warns_about_unmapped_or_dead_classes(caplog):
    with caplog.at_level(logging.WARNING):
        ThreatContextGenerator(MAPPING_PATH, known_classes=MODEL_CLASSES + ["Some_Future_Attack"])
    messages = " ".join(record.message for record in caplog.records)
    assert "Some_Future_Attack" in messages


def test_mapping_covers_exactly_the_model_classes():
    mapping = yaml.safe_load(open(MAPPING_PATH, encoding="utf-8"))
    assert set(mapping) == set(MODEL_CLASSES)


def test_every_mapped_playbook_is_whitelisted_and_within_action_limit():
    mapping = yaml.safe_load(open(MAPPING_PATH, encoding="utf-8"))
    policy = load_config("configs/trust_policy.yaml")
    edge_config = load_config("configs/edge_iiot.yaml")
    whitelist = set(policy["capability_whitelist"]["low_risk"]) | set(policy["capability_whitelist"]["high_risk"])
    max_actions = edge_config["trust_boundary"]["max_actions_per_plan"]

    for attack_type, entry in mapping.items():
        playbooks = entry["allowed_playbooks"]
        assert len(playbooks) <= max_actions, f"{attack_type}: {len(playbooks)} playbooks exceeds max_actions_per_plan={max_actions}"
        unknown = set(playbooks) - whitelist
        assert not unknown, f"{attack_type}: playbooks not in trust_policy.yaml's whitelist: {unknown}"
        if attack_type != "Normal":
            assert set(playbooks) & set(policy["capability_whitelist"]["low_risk"]), (
                f"{attack_type}: no low_risk playbook - would execute nothing below autonomous_execution trust"
            )
