from tfacd.runtime.contracts import IDSAlert
from tfacd.runtime.threat_context import ThreatContextGenerator

MAPPING_PATH = "configs/threat_context.yaml"


def test_known_attack_type_maps_severity_and_playbooks():
    generator = ThreatContextGenerator(MAPPING_PATH)
    context = generator.enrich(IDSAlert(attack_type="Port_Scanning", confidence=0.8))
    assert context.severity == "medium"
    assert context.priority == "P2"
    assert "block_source" in context.allowed_playbooks


def test_unknown_attack_type_falls_back_to_normal():
    generator = ThreatContextGenerator(MAPPING_PATH)
    context = generator.enrich(IDSAlert(attack_type="Totally_Unmapped_Type", confidence=0.5))
    assert context.severity == "informational"
    assert context.allowed_playbooks == ["observe", "log_event"]
