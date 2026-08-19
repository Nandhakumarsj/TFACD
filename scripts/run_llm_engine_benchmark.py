"""Real on-box benchmark gate for the LLM-backed decision engine (Ollama + a local
model). Must be run - and must report go=True - before configuring
agentic.decision_engine.engine: "llm" in configs/edge_iiot.yaml; see
agentic/factory.py::build_decision_engine, which refuses to build the LLM engine
without a passing report at the configured benchmark_report_path.

Pass --fallback-model (or set agentic.llm.fallback_model in the config) to also
benchmark a second, model-level fallback in the same run - the report becomes
{"primary": {...}, "fallback": {...}} instead of the flat single-model shape,
so a second `--model` invocation can no longer silently clobber the first
model's result at the same output path. factory.py reads both shapes.

Requires a running `ollama serve` with the target model(s) already pulled
(`ollama pull <model>`) - this script does not install or start Ollama itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_ollama import ChatOllama

from tfacd.agentic.benchmark import DEFAULT_TOTAL_VRAM_MB, BenchmarkResult, build_sample_contexts, nvidia_smi_total_vram_mb, run_benchmark
from tfacd.common.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--model", default=None, help="overrides agentic.llm.model from the config")
parser.add_argument(
    "--fallback-model", default=None,
    help="also benchmark a second, model-level fallback in this run (overrides agentic.llm.fallback_model from the "
    "config). Uses the same --base-url/--structured-output-method as the primary model - this is for a tool-calling "
    "fallback, not a json_mode escape hatch. Omit to keep today's single-model report shape.",
)
parser.add_argument("--base-url", default=None, help="overrides agentic.llm.base_url from the config")
parser.add_argument("--max-attempts", type=int, default=None, help="overrides agentic.llm.max_attempts from the config")
parser.add_argument(
    "--structured-output-method", default=None, choices=["function_calling", "json_mode"],
    help="overrides agentic.llm.structured_output_method from the config. 'function_calling' (default) requires the "
    "model to support Ollama tool calling - verified live: gemma3:4b returns HTTP 400 under this method. 'json_mode' "
    "works with any model but needs the schema spelled out in the prompt instead of via tool metadata.",
)
parser.add_argument(
    "--total-vram-mb", type=int, default=None,
    help="defaults to auto-detecting the installed card's total VRAM via nvidia-smi (works for a P4000, a P5000, or "
    "anything else) - pass this explicitly only to override that detection.",
)
parser.add_argument("--output", default=None, help="overrides agentic.llm.benchmark_report_path from the config")
args = parser.parse_args()

config = load_config(args.config)
llm_cfg = config.get("agentic", {}).get("llm", {})

model = args.model or llm_cfg.get("model", "qwen3:8b")
fallback_model = args.fallback_model or llm_cfg.get("fallback_model")
base_url = args.base_url or llm_cfg.get("base_url", "http://localhost:11434")
max_attempts = args.max_attempts if args.max_attempts is not None else int(llm_cfg.get("max_attempts", 2))
structured_output_method = args.structured_output_method or llm_cfg.get("structured_output_method", "function_calling")
output_path = Path(args.output or llm_cfg.get("benchmark_report_path", "artifacts/agentic/llm_benchmark_report.json"))

if args.total_vram_mb is not None:
    total_vram_mb = args.total_vram_mb
else:
    detected = nvidia_smi_total_vram_mb()
    if detected is None:
        print(f"Could not auto-detect total VRAM via nvidia-smi - falling back to DEFAULT_TOTAL_VRAM_MB={DEFAULT_TOTAL_VRAM_MB} "
              "(this project's own P4000). Pass --total-vram-mb explicitly if that's wrong for your card.")
        total_vram_mb = DEFAULT_TOTAL_VRAM_MB
    else:
        total_vram_mb = detected
        print(f"Detected {total_vram_mb} MB total VRAM via nvidia-smi.")

samples = build_sample_contexts(config["runtime"]["threat_context_mapping"])


def _benchmark_one(model_name: str) -> BenchmarkResult:
    chat_model = ChatOllama(
        model=model_name, base_url=base_url, temperature=float(llm_cfg.get("temperature", 0.0)), num_ctx=int(llm_cfg.get("num_ctx", 8192)),
        format="json" if structured_output_method == "json_mode" else None,
    )
    print(f"Benchmarking model={model_name!r} (structured_output_method={structured_output_method!r}) against {len(samples)} "
          f"representative threat contexts (base_url={base_url})...")
    return run_benchmark(
        chat_model, samples, model_name=model_name, max_attempts=max_attempts, total_vram_mb=total_vram_mb,
        structured_output_method=structured_output_method,
    )


def _print_result(label: str, result: BenchmarkResult) -> None:
    print()
    print(f"[{label}] model:                {result.model}")
    print(f"[{label}] vram_before_mb:        {result.vram_before_mb}")
    print(f"[{label}] vram_after_mb:         {result.vram_after_mb}")
    print(f"[{label}] vram_delta_mb:         {result.vram_delta_mb}")
    print(f"[{label}] tokens_per_second:     {result.tokens_per_second}")
    print(f"[{label}] latency_p50_seconds:   {result.latency_p50_seconds:.2f}")
    print(f"[{label}] latency_p95_seconds:   {result.latency_p95_seconds:.2f}")
    print(f"[{label}] raw_validity_rate:     {result.raw_validity_rate:.2f}")
    print(f"[{label}] retry_validity_rate:   {result.retry_validity_rate:.2f}")
    print(f"[{label}] go_no_go:              {result.go_no_go}")


primary_result = _benchmark_one(model)
_print_result("primary", primary_result)

output_path.parent.mkdir(parents=True, exist_ok=True)

if fallback_model:
    fallback_result = _benchmark_one(fallback_model)
    _print_result("fallback", fallback_result)
    report = {"primary": primary_result.to_dict(), "fallback": fallback_result.to_dict()}
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved dual-model report (primary={model!r}, fallback={fallback_model!r}): {output_path.resolve()}")
    if not primary_result.go_no_go["go"]:
        print("\nNOT READY: primary model failed its gate - fix the reasons above and re-run.")
    if not fallback_result.go_no_go["go"]:
        print("\nfallback model failed its gate - it will not be wired in as a fallback until this passes (primary-only operation still works).")
else:
    # Unchanged flat shape - exactly what factory.py has always read.
    output_path.write_text(json.dumps(primary_result.to_dict(), indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path.resolve()}")
    if not primary_result.go_no_go["go"]:
        print()
        print("NOT READY for agentic.decision_engine.engine: \"llm\" - fix the reasons above (e.g. a smaller model: --model qwen3:4b) and re-run.")
