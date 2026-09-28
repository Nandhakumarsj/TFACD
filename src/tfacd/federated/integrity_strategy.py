"""Live FTIL-defended Flower strategy.

Gate 4 (scripts/run_integrity_benchmark.py) measured PCAClusterEMAFilter's
TPR/FPR under controlled attacks in a standalone harness before this wiring was
enabled - per this file's original seam contract, that measurement had to exist
first. Findings that shape the defaults below: the detector alone only caught
malicious clients ~33% of the time (needs several consistent rounds before its
EMA-smoothed trust crosses the reject threshold), while coordinate_median and
trimmed_mean stayed ~0.93-0.94 macro-F1 under every attack tested. So detection
filters what it structurally/statistically can, and the aggregator itself stays
robust as defense-in-depth for whatever slips past - hence trimmed_mean as the
default aggregation_method rather than plain weighted_mean.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.app import ArrayRecord, MetricRecord
from flwr.common import Message
from flwr.serverapp.strategy import FedProx

from tfacd.integrity.aggregation import coordinate_median, trimmed_mean, weighted_average
from tfacd.integrity.detector import PCAClusterEMAFilter
from tfacd.integrity.update_validation import validate_update
from tfacd.integrity.vectorize import flatten_delta


def _state_to_numpy(array_record: ArrayRecord) -> dict[str, np.ndarray]:
    return {key: value.cpu().numpy() for key, value in array_record.to_torch_state_dict().items()}


def _numpy_to_arrayrecord(state: dict[str, np.ndarray]) -> ArrayRecord:
    # np.asarray guards against 0-d scalars (e.g. BatchNorm's num_batches_tracked
    # survives aggregation.py's reductions as a bare numpy.int64, not an ndarray,
    # and torch.from_numpy rejects that) - same defensive wrap every other
    # consumer of these dicts (flatten_delta, validate_update, aggregation.py) already has.
    return ArrayRecord({key: torch.from_numpy(np.asarray(value)) for key, value in state.items()})


class IntegrityAwareStrategy(FedProx):
    """FedProx (proximal-mu=0.0 is FedAvg-equivalent) with FTIL wired into
    aggregate_train: structural validation -> PCA-cluster-EMA detection ->
    robust aggregation over accepted replies, with per-round trust evidence
    persisted to a JSONL log.
    """

    def __init__(
        self,
        *args: Any,
        max_abs_parameter: float = 1_000_000.0,
        max_update_norm_ratio: float = 10.0,
        pca_components: int = 5,
        cluster_method: str = "agglomerative",
        min_benign_fraction: float = 0.5,
        ema_alpha: float = 0.3,
        reject_below_trust: float = 0.35,
        aggregation_method: str = "trimmed_mean",
        trim_ratio: float = 0.2,
        min_security_quorum: int = 3,
        trust_log_path: str | Path = "artifacts/models/ftil_trust_log.jsonl",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_abs_parameter = max_abs_parameter
        self.max_update_norm_ratio = max_update_norm_ratio
        self.aggregation_method = aggregation_method
        self.trim_ratio = trim_ratio
        # REVIEW FIX (P0/P1 - security quorum bypass): server_app.py's
        # min_train_nodes/min_available_nodes can be as low as 2, which is
        # below PCAClusterEMAFilter's own "n_clients < 3 -> accept all"
        # threshold AND below the point where trim_ratio actually trims
        # anything (floor(2 * 0.2) == 0). That combination meant a shrunk
        # federation silently bypassed both the detector and the robust
        # aggregator at once. min_security_quorum is enforced independently,
        # in this strategy (not the detector, whose job is classification, not
        # quorum policy): below it, the round is refused rather than
        # aggregated with "accept everyone" semantics.
        self.min_security_quorum = min_security_quorum
        self.detector = PCAClusterEMAFilter(
            pca_components=pca_components,
            cluster_method=cluster_method,
            min_benign_fraction=min_benign_fraction,
            ema_alpha=ema_alpha,
            reject_below_trust=reject_below_trust,
        )
        self.trust_log_path = Path(trust_log_path)
        self._current_arrays: ArrayRecord | None = None

    def configure_train(self, server_round, arrays, config, grid):  # noqa: D102 - Flower base signature
        self._current_arrays = arrays  # aggregate_train needs this as the validation/vectorization reference
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)  # 1. materialize replies

        valid_replies: list[Message] = []
        rejected_error = 0
        for msg in replies:
            if msg.has_error():  # 2. reject Flower error replies
                rejected_error += 1
                continue
            valid_replies.append(msg)

        if not valid_replies:
            return None, MetricRecord({"ftil_accepted": 0, "ftil_rejected_error": rejected_error})

        assert self._current_arrays is not None
        reference = _state_to_numpy(self._current_arrays)

        client_ids: list[str] = []
        client_states: dict[str, dict[str, np.ndarray]] = {}
        client_weights: dict[str, int] = {}
        rejected_validation: list[str] = []

        for msg in valid_replies:  # 3. extract ArrayRecord + client ID
            # REVIEW FIX (P0/P1 - client identity spoofing): client_id used to
            # come from the CLIENT-SUPPLIED "client-metadata" field whenever
            # present, i.e. the server trusted whatever ID a client claimed to
            # be. A malicious client could impersonate another client's ID,
            # polluting that ID's EMA trust history and trust-log evidence, and
            # potentially colliding with it in the per-client dictionaries
            # below. msg.metadata.src_node_id is Flower's own authenticated
            # sender identity and is now the ONLY source of client_id used for
            # trust/aggregation bookkeeping. Client-supplied metadata may still
            # be read for descriptive/debug purposes elsewhere, but must never
            # define identity.
            client_id = str(msg.metadata.src_node_id)
            candidate = _state_to_numpy(msg.content["arrays"])
            num_examples = int(msg.content["metrics"]["num-examples"])

            validation = validate_update(  # 4. structural validation against current global state
                candidate, reference, max_abs_parameter=self.max_abs_parameter, max_update_norm_ratio=self.max_update_norm_ratio
            )
            if not validation.accepted:
                rejected_validation.append(client_id)
                continue
            client_ids.append(client_id)
            client_states[client_id] = candidate
            client_weights[client_id] = num_examples

        # REVIEW FIX (P0/P1 - security quorum bypass): enforced here, before the
        # detector ever runs, so a federation that has shrunk below the
        # meaningful-security threshold cannot fall through to "too few
        # clients for clustering -> accept all" and cannot silently degenerate
        # trimmed_mean into an unrobust plain mean. No global model update is
        # produced for this round when quorum isn't met - the aggregator is
        # deliberately unavailable rather than silently insecure.
        if len(client_ids) < self.min_security_quorum:
            self._log_trust_evidence(server_round, [], rejected_validation, [], {}, {}, None)
            return None, MetricRecord(
                {
                    "ftil_accepted": 0,
                    "ftil_rejected_validation": len(rejected_validation),
                    "ftil_rejected_error": rejected_error,
                    "ftil_insufficient_security_quorum": 1,
                    "ftil_security_quorum": self.min_security_quorum,
                    "ftil_available_clients": len(client_ids),
                }
            )

        accepted_ids = list(client_ids)
        rejected_detector: list[str] = []
        trust_scores: dict[str, float] = {}
        ood_scores: dict[str, float] = {}
        detection_metrics = None
        if client_ids:  # 5. vectorize deltas + apply detector (handles <3 clients by accepting all)
            vectors = np.stack([flatten_delta(client_states[c], reference) for c in client_ids])
            detection = self.detector.detect(client_ids, vectors)
            accepted_ids = [c for c, keep in zip(client_ids, detection.benign_mask) if keep]
            rejected_detector = [c for c, keep in zip(client_ids, detection.benign_mask) if not keep]
            trust_scores = dict(zip(client_ids, detection.trust_scores.tolist()))
            ood_scores = dict(zip(client_ids, detection.ood_scores.tolist()))
            detection_metrics = detection.metrics

        # 7. persist evidence - includes the per-client OOD/personalization
        # scores and the clustering technique/quality metrics that produced this
        # round's decision, not just the accept/reject outcome.
        self._log_trust_evidence(server_round, accepted_ids, rejected_validation, rejected_detector, trust_scores, ood_scores, detection_metrics)

        metrics_payload: dict[str, int | float | str] = {
            "ftil_accepted": len(accepted_ids),
            "ftil_rejected_validation": len(rejected_validation),
            "ftil_rejected_detector": len(rejected_detector),
            "ftil_rejected_error": rejected_error,
        }
        if detection_metrics is not None:
            # cluster_method is a string - Flower's MetricRecord only accepts
            # int/float/list[int]/list[float] (raises TypeError otherwise), so it
            # stays out of this payload; it's already in the JSONL trust log's
            # "detection" block every round, which has no such restriction.
            metrics_payload["ftil_explained_variance_ratio"] = detection_metrics.explained_variance_ratio
            metrics_payload["ftil_silhouette"] = detection_metrics.silhouette if detection_metrics.silhouette is not None else -1.0
            metrics_payload["ftil_max_ood_score"] = max(ood_scores.values()) if ood_scores else 0.0
        metrics = MetricRecord(metrics_payload)
        if not accepted_ids:
            return None, metrics

        states = [client_states[c] for c in accepted_ids]  # 6. aggregate only accepted replies
        weights = [client_weights[c] for c in accepted_ids]
        if self.aggregation_method == "coordinate_median":
            aggregated = coordinate_median(states)
        elif self.aggregation_method == "trimmed_mean":
            aggregated = trimmed_mean(states, self.trim_ratio)
        else:
            aggregated = weighted_average(states, weights)

        return _numpy_to_arrayrecord(aggregated), metrics  # 8. return ArrayRecord + MetricRecord

    def _log_trust_evidence(
        self,
        server_round: int,
        accepted_ids: list[str],
        rejected_validation: list[str],
        rejected_detector: list[str],
        trust_scores: dict[str, float],
        ood_scores: dict[str, float],
        detection_metrics,
    ) -> None:
        self.trust_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "round": server_round,
            "accepted": accepted_ids,
            "rejected_validation": rejected_validation,
            "rejected_detector": rejected_detector,
            # trust_scores: EMA-smoothed personalized per-client trust (0..1),
            # carries across rounds. ood_scores: this round's distance from the
            # cohort's robust center in PCA space, as a multiple of the cohort's
            # median distance (1.0 = typical, memoryless). detection: which
            # clustering technique produced both, plus its own quality signals
            # (explained variance, silhouette) - see integrity/detector.py's
            # DetectionMetrics/DetectionResult docstrings for the distinction.
            "trust_scores": trust_scores,
            "ood_scores": ood_scores,
            "detection": detection_metrics.to_dict() if detection_metrics is not None else None,
        }
        with self.trust_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
