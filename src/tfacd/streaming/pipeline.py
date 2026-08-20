"""Runs the certified IDS model over streaming feature vectors, producing
IDSAlerts.

Batch-oriented replay loop over a pull-based generator (sources.py's
pd.read_csv(chunksize=...)) - there is no producer to apply backpressure to
and no broker, so this deliberately has no queue/rate-limiter/offset-
checkpointing machinery. batch_size is the only throughput knob that exists:
transform cost is a measured, ~fixed 0.65s per StreamingFeatureExtractor.transform()
call regardless of batch size (the fitted OneHotEncoder's vocabulary size
dominates), so bigger batches are strictly better up to memory limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from tfacd.federated.common import device as default_device
from tfacd.federated.common import model_from_metadata
from tfacd.integrity.certification import verify_release
from tfacd.runtime.contracts import IDSAlert
from tfacd.streaming.features import StreamingFeatureExtractor
from tfacd.streaming.sources import RecordSource
from tfacd.trust_boundary.preprocessing import canonicalize


@dataclass
class StreamingStats:
    records_read: int = 0
    alerts_emitted: int = 0
    alerts_suppressed: int = 0
    read_seconds: float = 0.0
    transform_seconds: float = 0.0
    inference_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.read_seconds + self.transform_seconds + self.inference_seconds

    @property
    def records_per_second(self) -> float:
        return self.records_read / self.total_seconds if self.total_seconds > 0 else 0.0


class StreamingIDS:
    def __init__(
        self,
        extractor: StreamingFeatureExtractor,
        model: nn.Module,
        classes: list[str],
        *,
        device: torch.device | None = None,
        sequence_length: int = 1,
        batch_size: int = 1024,
        min_confidence: float = 0.0,
        emit_normal: bool = False,
    ):
        if sequence_length != 1:
            # Refused, not silently supported: Gate 1 found this dataset's
            # frame.time is non-monotonic (no authentic row ordering) and its
            # rows are grouped by class in contiguous blocks - any window
            # spanning consecutive rows would leak the label through position.
            # See data/sequences.py's docstring for the same constraint on the
            # batch training plane; this is the honest version of the
            # diagram's "Flow Generation" box for this dataset.
            raise NotImplementedError(
                f"sequence_length={sequence_length} is refused for this dataset - see the module docstring"
            )
        self.extractor = extractor
        self.device = device or default_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.classes = classes
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.emit_normal = emit_normal

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> StreamingIDS:
        cfg = config["streaming"]
        result = verify_release(
            cfg["model_path"],
            public_key_path=cfg.get("public_key_path", "artifacts/keys/certification_ed25519_public.pem"),
            require_signature=bool(cfg.get("require_signature", True)),
            require_certified_status=bool(cfg.get("require_certified_status", True)),
        )
        if not result.ok:
            raise RuntimeError(f"certified checkpoint verification failed for {cfg['model_path']}: {'; '.join(result.reasons)}")

        extractor = StreamingFeatureExtractor(config["data"]["output_dir"])
        model = model_from_metadata(config)
        payload = torch.load(cfg["model_path"], map_location="cpu", weights_only=True)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(state_dict)

        return cls(
            extractor, model, extractor.metadata["classes"],
            sequence_length=int(config["data"].get("sequence_length", 1)),
            batch_size=int(cfg.get("batch_size", 1024)),
            min_confidence=float(cfg.get("min_confidence", 0.0)),
            emit_normal=bool(cfg.get("emit_normal", False)),
        )

    def _predict_batch(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        windowed = features[:, None, :]  # [n, feature_dim] -> [n, 1, feature_dim]; sequence_length=1 case
        tensor = torch.from_numpy(windowed).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)
        confidences, predicted = probabilities.max(dim=1)
        return predicted.cpu().numpy(), confidences.cpu().numpy()

    def _alert_for(self, record: dict, attack_type: str, confidence: float) -> IDSAlert | None:
        if (attack_type == "Normal" and not self.emit_normal) or confidence < self.min_confidence:
            return None
        # source_id/target_asset are raw, attacker-influenceable record fields (see
        # comment below) that flow unmodified into the LLM decision engine's prompt
        # (agentic/graph.py::build_human_prompt) and the semantic risk scorer
        # (trust_boundary/semantic_risk.py) - canonicalized here, once, at the single
        # point every downstream consumer reads from, rather than patching each
        # consumer separately. Unlike an agent's own plan parameters, this is upstream
        # sensor data, not a proposal to reject - so only normalize (NFKC + strip
        # zero-width/format chars), never reject on an obfuscation-looking pattern,
        # which would risk silently dropping a real attack alert.
        raw_source_id = record.get("ip.src_host")
        raw_target_asset = record.get("ip.dst_host")
        return IDSAlert(
            attack_type=attack_type,
            confidence=confidence,  # uncalibrated max softmax - no temperature scaling was performed
            source_id=canonicalize(raw_source_id) if raw_source_id is not None else None,  # dropped as a MODEL FEATURE (leakage), retained here as alert provenance
            target_asset=canonicalize(raw_target_asset) if raw_target_asset is not None else None,
            protocol=None,  # no informative protocol column at useful coverage - measured (mqtt.protoname ~6%, everything else ~0%), not guessed
        )

    def run(self, source: RecordSource) -> tuple[list[IDSAlert], StreamingStats]:
        stats = StreamingStats()
        alerts: list[IDSAlert] = []
        batch: list[dict] = []

        def flush() -> None:
            if not batch:
                return
            t0 = time.perf_counter()
            features = self.extractor.transform(batch)
            stats.transform_seconds += time.perf_counter() - t0

            t1 = time.perf_counter()
            predicted, confidences = self._predict_batch(features)
            stats.inference_seconds += time.perf_counter() - t1

            for record, class_idx, confidence in zip(batch, predicted, confidences):
                alert = self._alert_for(record, self.classes[int(class_idx)], float(confidence))
                if alert is None:
                    stats.alerts_suppressed += 1
                else:
                    alerts.append(alert)
                    stats.alerts_emitted += 1
            batch.clear()

        try:
            t_read = time.perf_counter()
            for record in source.records():
                stats.read_seconds += time.perf_counter() - t_read
                stats.records_read += 1
                batch.append(record)
                if len(batch) >= self.batch_size:
                    flush()
                t_read = time.perf_counter()
            flush()
        except KeyboardInterrupt:
            flush()
        return alerts, stats
