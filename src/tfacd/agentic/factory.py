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


def _require_passing_benchmark(report_path: str) -> None:
    path = Path(report_path)
    if not path.exists():
        raise RuntimeError(
            f"agentic.decision_engine.engine='llm' requires a passing benchmark report at {report_path}, "
            "but none exists - run scripts/run_llm_engine_benchmark.py first."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    go_no_go = report.get("go_no_go", {})
    if not go_no_go.get("go", False):
        reasons = "; ".join(go_no_go.get("reasons", [])) or "no reasons recorded"
        raise RuntimeError(
            f"agentic.decision_engine.engine='llm' but the benchmark report at {report_path} says go=False: {reasons} - "
            "re-run scripts/run_llm_engine_benchmark.py (optionally with a smaller --model) until it passes."
        )


def build_decision_engine(config: dict[str, Any], history: EntityHistory) -> DecisionEngine:
    agentic_cfg = config.get("agentic", {})
    engine_name = agentic_cfg.get("decision_engine", {}).get("engine", "template")

    if engine_name == "template":
        return AgenticDecisionEngine(history=history)

    if engine_name == "llm":
        llm_cfg = agentic_cfg.get("llm", {})
        if llm_cfg.get("require_benchmark_gate", True):
            _require_passing_benchmark(llm_cfg.get("benchmark_report_path", DEFAULT_BENCHMARK_REPORT_PATH))

        from langchain_ollama import ChatOllama

        from tfacd.agentic.llm_engine import LLMDecisionEngine

        # "function_calling" (default) requires the model to support Ollama tool
        # calling - verified live: gemma3:4b returns HTTP 400 "does not support
        # tools" under this method. "json_mode" works with any model but needs
        # format="json" set here at construction time (graph.py's _reason_node
        # can't set this on an already-built chat_model) plus the schema spelled
        # out in the prompt (handled in graph.py).
        structured_output_method = llm_cfg.get("structured_output_method", "function_calling")
        chat_model = ChatOllama(
            model=llm_cfg.get("model", "qwen3:8b"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            num_ctx=int(llm_cfg.get("num_ctx", 8192)),
            format="json" if structured_output_method == "json_mode" else None,
        )
        return LLMDecisionEngine(
            chat_model,
            history=history,
            fallback_engine=AgenticDecisionEngine(history=history),
            max_attempts=int(llm_cfg.get("max_attempts", 2)),
            structured_output_method=structured_output_method,
        )

    raise ValueError(f"unknown agentic.decision_engine.engine={engine_name!r} (expected 'template' or 'llm')")
