"""Config-driven selection between the deterministic and LLM-backed decision engines.

`ChatOllama`/`LLMDecisionEngine` are imported locally, inside the "llm" branch only -
same precedent as certification.py's local import of verify_file - so importing this
module (and therefore both demo scripts) never requires the `agentic-llm` optional
dependency group when the default "template" engine is configured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tfacd.agentic.base import DecisionEngine
from tfacd.agentic.decision_engine import AgenticDecisionEngine
from tfacd.agentic.history import EntityHistory

DEFAULT_BENCHMARK_REPORT_PATH = "artifacts/agentic/llm_benchmark_report.json"


def _load_benchmark_report(report_path: str) -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        raise RuntimeError(
            f"agentic.decision_engine.engine='llm' requires a passing benchmark report at {report_path}, "
            "but none exists - run scripts/run_llm_engine_benchmark.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _require_go(section: dict[str, Any], report_path: str, label: str) -> None:
    go_no_go = section.get("go_no_go", {})
    if not go_no_go.get("go", False):
        reasons = "; ".join(go_no_go.get("reasons", [])) or "no reasons recorded"
        raise RuntimeError(
            f"{label} benchmark report at {report_path} says go=False: {reasons} - "
            "re-run scripts/run_llm_engine_benchmark.py (optionally with a smaller --model) until it passes."
        )


def _build_chat_model(llm_cfg: dict[str, Any], model_name: str, structured_output_method: str):
    from langchain_ollama import ChatOllama

    # "function_calling" (default) requires the model to support Ollama tool
    # calling - verified live: gemma3:4b returns HTTP 400 "does not support
    # tools" under this method. "json_mode" works with any model but needs
    # format="json" set here at construction time (graph.py's _reason_node
    # can't set this on an already-built chat_model) plus the schema spelled
    # out in the prompt (handled in graph.py).
    return ChatOllama(
        model=model_name,
        base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        num_ctx=int(llm_cfg.get("num_ctx", 8192)),
        format="json" if structured_output_method == "json_mode" else None,
    )


def build_decision_engine(config: dict[str, Any], history: EntityHistory) -> DecisionEngine:
    agentic_cfg = config.get("agentic", {})
    engine_name = agentic_cfg.get("decision_engine", {}).get("engine", "template")

    if engine_name == "template":
        return AgenticDecisionEngine(history=history)

    if engine_name == "llm":
        llm_cfg = agentic_cfg.get("llm", {})
        report_path = llm_cfg.get("benchmark_report_path", DEFAULT_BENCHMARK_REPORT_PATH)
        fallback_model_name = llm_cfg.get("fallback_model")
        require_gate = llm_cfg.get("require_benchmark_gate", True)

        fallback_report: dict[str, Any] | None = None
        if require_gate:
            report = _load_benchmark_report(report_path)
            if "primary" in report:
                # New (dual-model) report shape, written by
                # run_llm_engine_benchmark.py --fallback-model.
                _require_go(report["primary"], report_path, "primary")
                fallback_report = report.get("fallback")
            else:
                # Old (single-model) report shape - unchanged, still supported.
                _require_go(report, report_path, "primary")

        from tfacd.agentic.llm_engine import LLMDecisionEngine

        structured_output_method = llm_cfg.get("structured_output_method", "function_calling")
        chat_model = _build_chat_model(llm_cfg, llm_cfg.get("model", "qwen3:8b"), structured_output_method)

        fallback_chat_model = None
        if fallback_model_name:
            if require_gate:
                if fallback_report is None:
                    raise RuntimeError(
                        f"agentic.llm.fallback_model={fallback_model_name!r} is configured but the benchmark report "
                        f"at {report_path} has no passing 'fallback' section - run "
                        f"scripts/run_llm_engine_benchmark.py --fallback-model {fallback_model_name} first."
                    )
                # A fallback that never passed its own gate is not silently
                # wired in just because it's configured.
                _require_go(fallback_report, report_path, "fallback")
            fallback_chat_model = _build_chat_model(llm_cfg, fallback_model_name, structured_output_method)

        return LLMDecisionEngine(
            chat_model,
            history=history,
            fallback_engine=AgenticDecisionEngine(history=history),
            fallback_chat_model=fallback_chat_model,
            max_attempts=int(llm_cfg.get("max_attempts", 2)),
            structured_output_method=structured_output_method,
        )

    raise ValueError(f"unknown agentic.decision_engine.engine={engine_name!r} (expected 'template' or 'llm')")
