import subprocess
from unittest.mock import patch

import pytest

from tfacd.agentic.benchmark import (
    build_sample_contexts,
    evaluate_go_no_go,
    nvidia_smi_memory_used_mb,
    nvidia_smi_total_vram_mb,
    run_benchmark,
)
from tfacd.agentic.graph import LLMProposedAction, LLMProposedPlan


def test_build_sample_contexts_covers_all_15_real_classes():
    samples = build_sample_contexts("configs/threat_context.yaml")
    assert len(samples) == 15
    class_names = {alert.attack_type for alert, _ in samples}
    assert len(class_names) == 15


def test_nvidia_smi_memory_used_mb_parses_output():
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="1234\n")
    with patch("subprocess.run", return_value=fake_result):
        assert nvidia_smi_memory_used_mb() == 1234


def test_nvidia_smi_memory_used_mb_returns_none_when_command_fails():
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        assert nvidia_smi_memory_used_mb() is None


def test_nvidia_smi_total_vram_mb_parses_output():
    # A P5000 (16GB) reporting its own total, not this project's P4000 (8GB) -
    # proves the gate isn't hardcoded to one card's number.
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="16384\n")
    with patch("subprocess.run", return_value=fake_result):
        assert nvidia_smi_total_vram_mb() == 16384


def test_nvidia_smi_total_vram_mb_returns_none_when_command_fails():
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        assert nvidia_smi_total_vram_mb() is None


def _baseline_go_no_go_kwargs():
    return dict(
        vram_before_mb=3000, vram_delta_mb=2000, tokens_per_second=15.0,
        latency_p95_seconds=5.0, raw_validity_rate=0.9, retry_validity_rate=1.0,
    )


def test_evaluate_go_no_go_passes_when_all_thresholds_met():
    result = evaluate_go_no_go(**_baseline_go_no_go_kwargs())
    assert result["go"] is True
    assert result["reasons"] == []


def test_evaluate_go_no_go_fails_on_vram_overflow():
    kwargs = _baseline_go_no_go_kwargs()
    kwargs["vram_delta_mb"] = 8000  # way more than 85% of (8192 - 3000) available
    result = evaluate_go_no_go(**kwargs)
    assert result["go"] is False
    assert any("vram_delta_mb" in reason for reason in result["reasons"])


def test_evaluate_go_no_go_fails_when_vram_unmeasurable():
    kwargs = _baseline_go_no_go_kwargs()
    kwargs["vram_before_mb"] = None
    kwargs["vram_delta_mb"] = None
    result = evaluate_go_no_go(**kwargs)
    assert result["go"] is False
    assert any("could not be measured" in reason for reason in result["reasons"])


def test_evaluate_go_no_go_fails_on_low_tokens_per_second():
    kwargs = _baseline_go_no_go_kwargs()
    kwargs["tokens_per_second"] = 1.0
    result = evaluate_go_no_go(**kwargs)
    assert result["go"] is False
    assert any("tokens_per_second" in reason for reason in result["reasons"])


def test_evaluate_go_no_go_fails_on_high_latency():
    kwargs = _baseline_go_no_go_kwargs()
    kwargs["latency_p95_seconds"] = 120.0
    result = evaluate_go_no_go(**kwargs)
    assert result["go"] is False
    assert any("latency_p95_seconds" in reason for reason in result["reasons"])


def test_measured_qwen3_8b_latency_passes_the_revised_ceiling():
    """The real p95 measured on this project's P4000 (40.2s) must pass, and the
    ceiling must still be a real gate - 90s is not acceptable."""
    kwargs = _baseline_go_no_go_kwargs()
    kwargs["latency_p95_seconds"] = 40.2
    assert evaluate_go_no_go(**kwargs)["go"] is True

    kwargs["latency_p95_seconds"] = 90.0
    assert evaluate_go_no_go(**kwargs)["go"] is False


def test_evaluate_go_no_go_fails_on_low_validity():
    kwargs = _baseline_go_no_go_kwargs()
    kwargs["raw_validity_rate"] = 0.1
    kwargs["retry_validity_rate"] = 0.2
    result = evaluate_go_no_go(**kwargs)
    assert result["go"] is False
    assert any("raw_validity_rate" in reason for reason in result["reasons"])
    assert any("retry_validity_rate" in reason for reason in result["reasons"])


class _FakeStructuredRunnable:
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        return self._responses.pop(0)


class _FakeChatModel:
    """Supports both plain structured output (used by the validity/latency pass
    in run_benchmark) and include_raw=True (used by measure_tokens_per_second)."""

    def __init__(self, plan_responses, raw_response=None):
        self._plan_responses = list(plan_responses)
        self._raw_response = raw_response

    def with_structured_output(self, schema, include_raw=False):
        if include_raw:
            return _FakeStructuredRunnable([self._raw_response] * 10)
        return _FakeStructuredRunnable(self._plan_responses)


def test_run_benchmark_computes_validity_and_latency_without_real_ollama():
    samples = build_sample_contexts("configs/threat_context.yaml")
    # one valid plan per sample, in call order - each proposes that sample's own
    # first allowed playbook so every one of the 15 validates successfully
    plans = [
        LLMProposedPlan(rationale="reasoned response", actions=[LLMProposedAction(capability=context.allowed_playbooks[0])], confidence=0.5)
        for _, context in samples
    ]
    chat_model = _FakeChatModel(plans, raw_response={"raw": None, "parsed": plans[0]})

    with patch("tfacd.agentic.benchmark.nvidia_smi_memory_used_mb", return_value=1000):
        result = run_benchmark(chat_model, samples, model_name="fake-model", max_attempts=2)

    assert result.num_samples == 15
    assert result.raw_validity_rate == 1.0
    assert result.retry_validity_rate == 1.0
    assert result.vram_before_mb == 1000
    assert result.tokens_per_second is None  # raw_response's "raw" is None -> no response_metadata to read
