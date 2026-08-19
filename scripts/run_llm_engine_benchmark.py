"""Real on-box benchmark gate for the LLM-backed decision engine (Ollama + a local
model). Must be run - and must report go=True - before configuring
agentic.decision_engine.engine: "llm" in configs/edge_iiot.yaml; see
agentic/factory.py::build_decision_engine, which refuses to build the LLM engine
without a passing report at the configured benchmark_report_path.

Requires a running `ollama serve` with the target model already pulled
(`ollama pull <model>`) - this script does not install or start Ollama itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_ollama import ChatOllama

from tfacd.agentic.benchmark import DEFAULT_TOTAL_VRAM_MB, build_sample_contexts, nvidia_smi_total_vram_mb, run_benchmark
from tfacd.common.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--model", default=None, help="overrides agentic.llm.model from the config")
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

chat_model = ChatOllama(
    model=model, base_url=base_url, temperature=float(llm_cfg.get("temperature", 0.0)), num_ctx=int(llm_cfg.get("num_ctx", 8192)),
    format="json" if structured_output_method == "json_mode" else None,
)
samples = build_sample_contexts(config["runtime"]["threat_context_mapping"])

print(f"Benchmarking model={model!r} (structured_output_method={structured_output_method!r}) against {len(samples)} "
      f"representative threat contexts (base_url={base_url})...")
result = run_benchmark(
    chat_model, samples, model_name=model, max_attempts=max_attempts, total_vram_mb=total_vram_mb,
    structured_output_method=structured_output_method,
)

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

print()
print(f"vram_before_mb:       {result.vram_before_mb}")
print(f"vram_after_mb:        {result.vram_after_mb}")
print(f"vram_delta_mb:        {result.vram_delta_mb}")
print(f"tokens_per_second:    {result.tokens_per_second}")
print(f"latency_p50_seconds:  {result.latency_p50_seconds:.2f}")
print(f"latency_p95_seconds:  {result.latency_p95_seconds:.2f}")
print(f"raw_validity_rate:    {result.raw_validity_rate:.2f}")
print(f"retry_validity_rate:  {result.retry_validity_rate:.2f}")
print()
print(f"go_no_go: {result.go_no_go}")
print(f"Saved: {output_path.resolve()}")

if not result.go_no_go["go"]:
    print()
    print("NOT READY for agentic.decision_engine.engine: \"llm\" - fix the reasons above (e.g. a smaller model: --model qwen3:4b) and re-run.")
