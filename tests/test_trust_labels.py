import json

from tfacd.analytics.trust_labels import AnalystLabel, AnalystLabelStore, verify_label_chain


def make_label(audit_sequence, label="correct", analyst_id="alice"):
    return AnalystLabel(audit_sequence=audit_sequence, label=label, analyst_id=analyst_id, rationale="looks right")


def test_append_and_load_all_round_trips(tmp_path):
    path = tmp_path / "labels.jsonl"
    store = AnalystLabelStore(path)
    store.append(make_label(1, "correct"))
    store.append(make_label(2, "false_positive"))

    loaded = store.load_all()
    assert [label.audit_sequence for label in loaded] == [1, 2]
    assert [label.label for label in loaded] == ["correct", "false_positive"]


def test_for_sequence_filters_to_matching_audit_sequence(tmp_path):
    store = AnalystLabelStore(tmp_path / "labels.jsonl")
    store.append(make_label(5, "correct", "alice"))
    store.append(make_label(5, "wrong_trust_level", "bob"))
    store.append(make_label(6, "correct", "alice"))

    matches = store.for_sequence(5)
    assert {label.analyst_id for label in matches} == {"alice", "bob"}
    assert all(label.audit_sequence == 5 for label in matches)


def test_chain_verifies_when_untampered(tmp_path):
    path = tmp_path / "labels.jsonl"
    store = AnalystLabelStore(path)
    for i in range(5):
        store.append(make_label(i))

    ok, bad_index = verify_label_chain(path)
    assert ok
    assert bad_index is None


def test_tamper_is_detected(tmp_path):
    path = tmp_path / "labels.jsonl"
    store = AnalystLabelStore(path)
    for i in range(5):
        store.append(make_label(i))

    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[2])
    tampered["label"]["label"] = "false_negative"
    lines[2] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad_index = verify_label_chain(path)
    assert not ok
    assert bad_index == 2


def test_verify_label_chain_on_missing_file_is_ok(tmp_path):
    ok, bad_index = verify_label_chain(tmp_path / "does_not_exist.jsonl")
    assert ok
    assert bad_index is None


def test_store_resumes_chain_across_instances(tmp_path):
    path = tmp_path / "labels.jsonl"
    AnalystLabelStore(path).append(make_label(1))

    second = AnalystLabelStore(path)
    second.append(make_label(2))

    ok, _ = verify_label_chain(path)
    assert ok
    assert len(second.load_all()) == 2
