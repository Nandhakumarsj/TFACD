from datetime import timedelta

from tfacd.agentic.history import EntityHistory


def test_in_memory_recent_and_count():
    history = EntityHistory()
    history.append("agent-1", "incident", {"attack_type": "DDoS_HTTP"})
    history.append("agent-1", "trust_decision", {"accepted": False})
    history.append("agent-2", "incident", {"attack_type": "Normal"})

    assert len(history.recent("agent-1")) == 2
    assert len(history.recent("agent-1", kind="incident")) == 1
    assert history.count_since("agent-1", within=timedelta(hours=1)) == 2
    assert history.recent("unknown-agent") == []


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "history.jsonl"
    first = EntityHistory(persist_path=path)
    first.append("agent-1", "incident", {"attack_type": "Port_Scanning"})

    second = EntityHistory(persist_path=path)
    assert len(second.recent("agent-1")) == 1
    second.append("agent-1", "trust_decision", {"accepted": True})

    third = EntityHistory(persist_path=path)
    assert len(third.recent("agent-1")) == 2
