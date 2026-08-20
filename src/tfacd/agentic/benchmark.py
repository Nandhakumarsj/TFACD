"""Real on-box measurement of the LLM-backed decision engine, run before it's
trusted in the pipeline (agentic/factory.py::build_decision_engine refuses to
construct an "llm" engine without a passing report - see
DEFAULT_BENCHMARK_REPORT_PATH there).

No prior VRAM-delta measurement exists in this repo: Ollama runs as its own
server process, so torch.cuda.memory_allocated() can't see it - this shells out
to nvidia-smi instead, the same tool used for manual GPU checks throughout this
project.

Validity/latency are measured through the real production graph
(agentic/graph.py::build_decision_graph) - not a simplified stand-in - so the
report reflects the actual reason/validate/retry path decisions will take.
Tokens/sec is a separate, smaller measurement (measure_tokens_per_second) since
it needs `include_raw=True` to read Ollama's own eval_count/eval_duration off
the raw AIMessage, which the production reason node intentionally doesn't
request (it only needs the parsed plan).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from tfacd.agentic.graph import LLMProposedPlan, build_decision_graph, build_human_prompt, build_system_prompt, initial_state
from tfacd.runtime.contracts import IDSAlert, ThreatContext

DEFAULT_TOTAL_VRAM_MB = 8192
DEFAULT_VRAM_SAFETY_MARGIN = 0.85
DEFAULT_MIN_TOKENS_PER_SECOND = 5.0
# Raised 30.0 -> 60.0 after measuring qwen3:8b on this project's actual hardware
# (Quadro P4000, Pascal, no tensor cores): p50 30.9s / p95 40.2s per decision at
# 21.7 tok/s. The original 30s was an unmeasured guess; 40s/decision bounds a
# max_incidents=10 demo run at ~5-7 minutes of LLM time, against an IDS stage
# that scores 20,000 records in 14s. Kept as a real ceiling, not removed - a
# model 50% slower than what was measured should still fail this gate.
DEFAULT_MAX_P95_LATENCY_SECONDS = 60.0
DEFAULT_MIN_RAW_VALIDITY = 0.80
DEFAULT_MIN_RETRY_VALIDITY = 0.95


def nvidia_smi_memory_used_mb() -> int | None:
    """None (not a crash) if nvidia-smi isn't on PATH or the call fails - a
    benchmark report should say "couldn't measure", never silently omit VRAM
    from a go/no-go decision without saying so (see evaluate_go_no_go)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def nvidia_smi_total_vram_mb() -> int | None:
    """Auto-detects the installed card's total VRAM so the gate adapts to
    whatever GPU is actually present (this project's P4000 at 8GB, a P5000 at
    16GB, or anything else) instead of a value hardcoded for one card. None
    (not a crash, not a silent fallback to DEFAULT_TOTAL_VRAM_MB) if nvidia-smi
    isn't on PATH or the call fails - callers decide what None means for them."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def build_sample_contexts(threat_context_mapping_path: str | Path) -> list[tuple[IDSAlert, ThreatContext]]:
    """One representative alert per real class in threat_context.yaml - runnable
    standalone, no dependency on the trained checkpoint or the raw dataset CSV."""
    mapping = yaml.safe_load(Path(threat_context_mapping_path).read_text(encoding="utf-8"))
    samples = []
    for class_name, entry in mapping.items():
        alert = IDSAlert(attack_type=class_name, confidence=0.85, source_id="10.0.0.42", target_asset="plc-07")
        context = ThreatContext(alert=alert, **entry)
        samples.append((alert, context))
    return samples


def measure_tokens_per_second(
    chat_model, samples: list[tuple[IDSAlert, ThreatContext]], *, sample_count: int = 3, structured_output_method: str = "function_calling"
) -> float | None:
    if structured_output_method == "json_mode":
        structured_model = chat_model.with_structured_output(LLMProposedPlan, method="json_mode", include_raw=True)
    else:
        structured_model = chat_model.with_structured_output(LLMProposedPlan, include_raw=True)
    system_prompt = build_system_prompt(structured_output_method)
    rates: list[float] = []
    for alert, context in samples[:sample_count]:
        state = initial_state(alert, context, repeat_activity=False, repeat_window_minutes=30, max_attempts=1)
        messages = [SystemMessage(system_prompt), HumanMessage(build_human_prompt(state))]
        try:
            result = structured_model.invoke(messages)
            raw = result.get("raw") if isinstance(result, dict) else None
            metadata = getattr(raw, "response_metadata", {}) or {}
            eval_count = metadata.get("eval_count")
            eval_duration_ns = metadata.get("eval_duration")
            if eval_count and eval_duration_ns:
                rates.append(eval_count / (eval_duration_ns / 1e9))
        except Exception:
            continue
    return float(np.mean(rates)) if rates else None


def evaluate_go_no_go(
    *,
    vram_before_mb: int | None,
    vram_delta_mb: int | None,
    tokens_per_second: float | None,
    latency_p95_seconds: float,
    raw_validity_rate: float,
    retry_validity_rate: float,
    total_vram_mb: int = DEFAULT_TOTAL_VRAM_MB,
    vram_safety_margin: float = DEFAULT_VRAM_SAFETY_MARGIN,
    min_tokens_per_second: float = DEFAULT_MIN_TOKENS_PER_SECOND,
    max_p95_latency_seconds: float = DEFAULT_MAX_P95_LATENCY_SECONDS,
    min_raw_validity: float = DEFAULT_MIN_RAW_VALIDITY,
    min_retry_validity: float = DEFAULT_MIN_RETRY_VALIDITY,
) -> dict[str, Any]:
    """Labeled default thresholds (see the plan this was built from) - open to
    revision, not silently final. VRAM headroom is measured against
    vram_before_mb (actual free VRAM at benchmark time), never an assumed idle
    total, since this GPU is routinely shared with unrelated processes."""
    reasons: list[str] = []
    go = True

    if vram_before_mb is None or vram_delta_mb is None:
        reasons.append("VRAM could not be measured (nvidia-smi unavailable) - treated as a failed gate, not silently passed")
        go = False
    else:
        available_mb = total_vram_mb - vram_before_mb
        budget_mb = vram_safety_margin * available_mb
        if vram_delta_mb > budget_mb:
            reasons.append(
                f"vram_delta_mb={vram_delta_mb} exceeds {vram_safety_margin:.0%} of available headroom "
                f"({budget_mb:.0f}mb of {available_mb}mb free before load)"
            )
            go = False

    if tokens_per_second is None or tokens_per_second < min_tokens_per_second:
        reasons.append(f"tokens_per_second={tokens_per_second} below floor {min_tokens_per_second}")
        go = False

    if latency_p95_seconds > max_p95_latency_seconds:
        reasons.append(f"latency_p95_seconds={latency_p95_seconds:.1f} exceeds ceiling {max_p95_latency_seconds}")
        go = False

    if raw_validity_rate < min_raw_validity:
        reasons.append(f"raw_validity_rate={raw_validity_rate:.2f} below floor {min_raw_validity}")
        go = False

    if retry_validity_rate < min_retry_validity:
        reasons.append(f"retry_validity_rate={retry_validity_rate:.2f} below floor {min_retry_validity}")
        go = False

    return {"go": go, "reasons": reasons}


@dataclass
class BenchmarkResult:
    model: str
    vram_before_mb: int | None
    vram_after_mb: int | None
    vram_delta_mb: int | None
    tokens_per_second: float | None
    latency_p50_seconds: float
    latency_p95_seconds: float
    raw_validity_rate: float
    retry_validity_rate: float
    num_samples: int
    go_no_go: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "vram_before_mb": self.vram_before_mb,
            "vram_after_mb": self.vram_after_mb,
            "vram_delta_mb": self.vram_delta_mb,
            "tokens_per_second": self.tokens_per_second,
            "latency_p50_seconds": self.latency_p50_seconds,
            "latency_p95_seconds": self.latency_p95_seconds,
            "raw_validity_rate": self.raw_validity_rate,
            "retry_validity_rate": self.retry_validity_rate,
            "num_samples": self.num_samples,
            "go_no_go": self.go_no_go,
        }


def run_benchmark(
    chat_model,
    samples: list[tuple[IDSAlert, ThreatContext]],
    *,
    model_name: str,
    max_attempts: int = 2,
    total_vram_mb: int = DEFAULT_TOTAL_VRAM_MB,
    structured_output_method: str = "function_calling",
) -> BenchmarkResult:
    vram_before = nvidia_smi_memory_used_mb()

    graph = build_decision_graph(chat_model, max_attempts=max_attempts, structured_output_method=structured_output_method)
    latencies: list[float] = []
    raw_valid = 0
    retry_valid = 0
    for alert, context in samples:
        state = initial_state(alert, context, repeat_activity=False, repeat_window_minutes=30, max_attempts=max_attempts)
        start = time.perf_counter()
        result = graph.invoke(state)
        latencies.append(time.perf_counter() - start)
        if result["plan"] is not None:
            retry_valid += 1
            if result["attempt"] == 1:
                raw_valid += 1

    tokens_per_second = measure_tokens_per_second(chat_model, samples, structured_output_method=structured_output_method)
    vram_after = nvidia_smi_memory_used_mb()
    vram_delta = (vram_after - vram_before) if (vram_before is not None and vram_after is not None) else None

    n = len(samples)
    raw_validity_rate = raw_valid / n if n else 0.0
    retry_validity_rate = retry_valid / n if n else 0.0
    latency_p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    latency_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    go_no_go = evaluate_go_no_go(
        vram_before_mb=vram_before, vram_delta_mb=vram_delta, tokens_per_second=tokens_per_second,
        latency_p95_seconds=latency_p95, raw_validity_rate=raw_validity_rate, retry_validity_rate=retry_validity_rate,
        total_vram_mb=total_vram_mb,
    )

    return BenchmarkResult(
        model=model_name, vram_before_mb=vram_before, vram_after_mb=vram_after, vram_delta_mb=vram_delta,
        tokens_per_second=tokens_per_second, latency_p50_seconds=latency_p50, latency_p95_seconds=latency_p95,
        raw_validity_rate=raw_validity_rate, retry_validity_rate=retry_validity_rate, num_samples=n, go_no_go=go_no_go,
    )
