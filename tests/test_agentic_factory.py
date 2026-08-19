import json

import pytest

from tfacd.agentic.decision_engine import AgenticDecisionEngine
from tfacd.agentic.factory import build_decision_engine
from tfacd.agentic.history import EntityHistory
from tfacd.agentic.llm_engine import LLMDecisionEngine


def test_absent_agentic_section_defaults_to_template():
    engine = build_decision_engine({}, EntityHistory())
    assert isinstance(engine, AgenticDecisionEngine)


def test_explicit_template_engine():
    config = {"agentic": {"decision_engine": {"engine": "template"}}}
    engine = build_decision_engine(config, EntityHistory())
    assert isinstance(engine, AgenticDecisionEngine)


def test_llm_engine_without_benchmark_report_raises(tmp_path):
    report_path = tmp_path / "missing_report.json"
    config = {
        "agentic": {
            "decision_engine": {"engine": "llm"},
            "llm": {"benchmark_report_path": str(report_path)},
        }
    }
    with pytest.raises(RuntimeError, match="benchmark report"):
        build_decision_engine(config, EntityHistory())


def test_llm_engine_with_failing_benchmark_report_raises(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"go_no_go": {"go": False, "reasons": ["vram insufficient"]}}), encoding="utf-8")
    config = {
        "agentic": {
            "decision_engine": {"engine": "llm"},
            "llm": {"benchmark_report_path": str(report_path)},
        }
    }
    with pytest.raises(RuntimeError, match="vram insufficient"):
        build_decision_engine(config, EntityHistory())


def test_llm_engine_with_passing_benchmark_report_builds_llm_engine(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"go_no_go": {"go": True, "reasons": []}}), encoding="utf-8")
    config = {
        "agentic": {
            "decision_engine": {"engine": "llm"},
            "llm": {"benchmark_report_path": str(report_path), "model": "qwen3:8b"},
        }
    }
    engine = build_decision_engine(config, EntityHistory())
    assert isinstance(engine, LLMDecisionEngine)


def test_llm_engine_skips_gate_when_require_benchmark_gate_is_false(tmp_path):
    config = {
        "agentic": {
            "decision_engine": {"engine": "llm"},
            "llm": {"require_benchmark_gate": False},
        }
    }
    engine = build_decision_engine(config, EntityHistory())
    assert isinstance(engine, LLMDecisionEngine)


def test_unknown_engine_name_raises():
    config = {"agentic": {"decision_engine": {"engine": "not-a-real-engine"}}}
    with pytest.raises(ValueError, match="unknown"):
        build_decision_engine(config, EntityHistory())
