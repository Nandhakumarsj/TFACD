"""Closest available analog to the Phase II diagram's "Feedback Learning ->
Adaptive Threshold Optimizer" box - but it tunes FTIL's PCAClusterEMAFilter
parameters (reject_below_trust, ema_alpha) against integrity/benchmark.py's
labeled attack/benign ground truth, NOT the agentic Trust Boundary's own
Rs/Rc/Rb/T thresholds. No ground truth for "this trust decision was actually
wrong" exists anywhere on the agentic side (see analytics/drift.py and
reputation.py's module docstrings for the same limitation), so there is
nothing to adapt those thresholds against. This module never touches
trust_boundary/dynamic_trust.py or trust_policy.yaml.

Second scope caveat: candidates here are scored by integrity/benchmark.py's
"pca_cluster_ema" mode, which aggregates detector-accepted clients with plain
weighted_average - i.e. this measures the PCA-cluster-EMA detector's TPR/FPR
IN ISOLATION. It does not evaluate the LIVE deployed strategy
(federated/integrity_strategy.py's IntegrityAwareStrategy), which aggregates
with trimmed_mean instead specifically because Gate 4 found trimmed_mean alone
already holds ~0.93-0.94 macro-F1 under every attack tested regardless of
detector threshold. So a gain found here is "improves the isolated detector's
TPR/FPR," not "improves live end-to-end robustness" - those are different
claims and this module only supports the first one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from tfacd.integrity.benchmark import run_benchmark

REJECT_BELOW_TRUST_GRID = [0.25, 0.35, 0.45]
EMA_ALPHA_GRID = [0.3, 0.5, 0.7]

# Gate 4's original published baseline (reject_below_trust=0.35, ema_alpha=0.3 -
# configs/edge_iiot.yaml's defaults, full 5-scenario/4-mode matrix). Hardcoded,
# not recomputed, per this session's earlier Gate 4 run.
BASELINE_TPR = 0.333
BASELINE_FPR = 0.0


@dataclass
class GridResult:
    params: dict[str, float]  # {"reject_below_trust": ..., "ema_alpha": ...}
    detection_metrics: dict[str, Any]  # run_benchmark's "detection_metrics" dict for this candidate


def run_grid_search(config: dict[str, Any], on_progress=None) -> list[GridResult]:
    """9 candidates (3x3), each a cheap narrow run_benchmark call: 2 rounds,
    sign_flip + gaussian_noise only, pca_cluster_ema only. Mutates a per-candidate
    copy of config["integrity"]; never mutates the caller's config in place.
    """
    results: list[GridResult] = []
    for reject_below_trust in REJECT_BELOW_TRUST_GRID:
        for ema_alpha in EMA_ALPHA_GRID:
            trial_config = copy.deepcopy(config)
            trial_config["integrity"]["reject_below_trust"] = reject_below_trust
            trial_config["integrity"]["ema_alpha"] = ema_alpha
            report = run_benchmark(
                trial_config,
                num_rounds=2,
                scenarios=["sign_flip", "gaussian_noise"],
                modes=["pca_cluster_ema"],
                max_samples_per_client=3000,
                init_checkpoint="artifacts/models/centralized_best.pt",
                seed=42,
            )
            result = GridResult({"reject_below_trust": reject_below_trust, "ema_alpha": ema_alpha}, report["detection_metrics"])
            results.append(result)
            if on_progress:
                on_progress(result)
    return results


def select_best_candidate(results: list[GridResult], fpr_tolerance: float = 0.1) -> tuple[GridResult, bool]:
    """Maximize tpr among candidates with fpr <= fpr_tolerance. If none qualify
    (or every candidate has an undefined tpr/fpr - e.g. a scenario contributed no
    malicious or no benign client, see benchmark.py's tp/(tp+fn)-is-None guard),
    falls back to maximizing (tpr - fpr) as a simple, legible tradeoff score
    across whatever candidates DO have both metrics defined - not a claim of
    multi-objective optimality, just "don't crash, report the closest thing".

    Returns (chosen, met_constraint) so callers can report which case happened.
    """
    scored = [r for r in results if r.detection_metrics.get("tpr") is not None and r.detection_metrics.get("fpr") is not None]
    if not scored:
        raise ValueError("no candidate produced a usable tpr/fpr (all undefined)")

    within_tolerance = [r for r in scored if r.detection_metrics["fpr"] <= fpr_tolerance]
    if within_tolerance:
        return max(within_tolerance, key=lambda r: r.detection_metrics["tpr"]), True
    return max(scored, key=lambda r: r.detection_metrics["tpr"] - r.detection_metrics["fpr"]), False


# --- AGENTIC ASTB ANALYST FEEDBACK & THRESHOLD OPTIMIZER ---

import json
from pathlib import Path


@dataclass
class AnalystFeedbackRecord:
    incident_id: str
    agent_id: str
    semantic_risk: float
    consistency: float
    behavioral_trust: float
    expected_accepted: bool
    label: str  # "correct", "false_positive", "false_negative", "over_restricted"
    notes: str = ""


class AnalystFeedbackStore:
    """JSONL-backed persistent store for human analyst ground-truth labels."""

    def __init__(self, store_path: str | Path = "artifacts/analytics/analyst_feedback.jsonl"):
        self.store_path = Path(store_path)

    def add_feedback(self, record: AnalystFeedbackRecord) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "incident_id": record.incident_id,
            "agent_id": record.agent_id,
            "semantic_risk": record.semantic_risk,
            "consistency": record.consistency,
            "behavioral_trust": record.behavioral_trust,
            "expected_accepted": record.expected_accepted,
            "label": record.label,
            "notes": record.notes,
        }
        with self.store_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def load_feedback(self) -> list[AnalystFeedbackRecord]:
        if not self.store_path.exists():
            return []
        records = []
        with self.store_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    records.append(AnalystFeedbackRecord(**d))
        return records


@dataclass
class AgenticGridResult:
    weights: tuple[float, float, float]  # (ws, wc, wb)
    thresholds: dict[str, float]  # {low, medium, high}
    f1_score: float
    precision: float
    recall: float


def run_agentic_grid_search(feedback_records: list[AnalystFeedbackRecord]) -> list[AgenticGridResult]:
    """Optimizes ASTB trust weights (ws, wc, wb) and threshold boundaries against ground-truth analyst feedback."""
    if not feedback_records:
        return []

    weight_candidates = [
        (0.4, 0.3, 0.3),
        (0.5, 0.3, 0.2),
        (0.3, 0.4, 0.3),
        (0.33, 0.33, 0.34),
    ]
    threshold_candidates = [
        {"low": 0.40, "medium": 0.65, "high": 0.85},
        {"low": 0.35, "medium": 0.60, "high": 0.80},
        {"low": 0.45, "medium": 0.70, "high": 0.90},
    ]

    results: list[AgenticGridResult] = []

    for ws, wc, wb in weight_candidates:
        for thresh in threshold_candidates:
            tp = fp = fn = tn = 0
            for rec in feedback_records:
                # T = ws * (1 - Rs) + wc * Rc + wb * Rb
                trust_val = ws * (1.0 - rec.semantic_risk) + wc * rec.consistency + wb * rec.behavioral_trust
                predicted_accepted = trust_val >= thresh["low"]

                if predicted_accepted and rec.expected_accepted:
                    tp += 1
                elif predicted_accepted and not rec.expected_accepted:
                    fp += 1
                elif not predicted_accepted and rec.expected_accepted:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append(AgenticGridResult(weights=(ws, wc, wb), thresholds=thresh, f1_score=f1, precision=precision, recall=recall))

    return sorted(results, key=lambda r: r.f1_score, reverse=True)
