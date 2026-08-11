"""Standalone FTIL attack/defense benchmark (Gate 4).

This intentionally does not go through Flower's ClientApp/ServerApp/simulation
engine. It is the "standalone integrity tests" referenced by
`federated/integrity_strategy.py`: measure detector TPR/FPR under controlled
attacks before wiring PCAClusterEMAFilter into the live training loop.

Client-round training data is capped (`max_samples_per_client`) purely to keep
this diagnostic benchmark's wall-clock bounded across the attack x defense
matrix; it does not affect Gate 2/3 training or the live federated pipeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tfacd.data.dataset import SequenceDataset
from tfacd.data.preprocess import load_prepared
from tfacd.data.sequences import make_sequences
from tfacd.federated.common import model_from_metadata
from tfacd.integrity.aggregation import coordinate_median, trimmed_mean, weighted_average
from tfacd.integrity.attacks import gaussian_noise, gradual_scaling, label_flip_to_normal, sign_flip
from tfacd.integrity.detector import PCAClusterEMAFilter
from tfacd.integrity.update_validation import validate_update
from tfacd.integrity.vectorize import flatten_delta
from tfacd.training.engine import evaluate as evaluate_epoch
from tfacd.training.engine import train_one_epoch

ATTACK_SCENARIOS = ["none", "label_flip", "sign_flip", "gaussian_noise", "gradual_scaling"]
AGGREGATION_MODES = ["no_defense", "pca_cluster_ema", "coordinate_median", "trimmed_mean"]


def _state_to_numpy(state_dict: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def _numpy_to_device_tensors(state: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in state.items()}


def _client_train_loader(prepared, output_dir, client_id, seq_len, stride, batch_size, max_samples, seed, label_transform=None):
    indices = np.load(Path(output_dir) / "partitions" / f"client_{client_id}.npy")
    if max_samples and len(indices) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_samples, replace=False)
    x = prepared.x_train[indices]
    y = prepared.y_train[indices]
    if label_transform is not None:
        y = label_transform(y)
    x_seq, y_seq = make_sequences(x, y, seq_len, stride)
    return DataLoader(SequenceDataset(x_seq, y_seq), batch_size=batch_size, shuffle=True)


@dataclass
class RoundLog:
    scenario: str
    mode: str
    round: int
    malicious_ids: list[int]
    rejected_ids: list[int]
    caught_by: dict[str, str]
    aggregation_seconds: float


def run_benchmark(
    config: dict[str, Any],
    num_rounds: int = 3,
    malicious_client_ids: tuple[int, ...] = (0,),
    max_samples_per_client: int = 15000,
    seed: int = 42,
    init_checkpoint: str | Path | None = None,
    on_progress=None,
    scenarios: list[str] | None = None,
    modes: list[str] | None = None,
) -> dict[str, Any]:
    # None (the default) preserves the original full-matrix behavior byte-for-byte;
    # a caller passing either list gets a narrower, cheaper run (e.g. feedback_loop's
    # grid search only needs sign_flip/gaussian_noise x pca_cluster_ema, not all 20 cells).
    run_scenarios = scenarios if scenarios is not None else ATTACK_SCENARIOS
    run_modes = modes if modes is not None else AGGREGATION_MODES
    data_cfg = config["data"]
    fed_cfg = config["federated"]
    integ_cfg = config["integrity"]
    output_dir = data_cfg["output_dir"]

    prepared = load_prepared(output_dir)
    metadata = json.loads((Path(output_dir) / "metadata.json").read_text(encoding="utf-8"))
    class_names = [str(c) for c in metadata["classes"]]
    normal_index = class_names.index("Normal") if "Normal" in class_names else 0

    seq_len = int(data_cfg.get("sequence_length", 1))
    stride = int(data_cfg.get("sequence_stride", 1))
    batch_size = int(config["training"]["batch_size"])
    num_clients = int(fed_cfg["num_clients"])
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test, y_test = make_sequences(prepared.x_test, prepared.y_test, seq_len, stride)
    test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    init_model = model_from_metadata(config).to(device)
    if init_checkpoint is not None:
        # Fine-tune-under-attack, not learn-from-scratch-under-attack: an already
        # competent global model is both the realistic FL threat model (poisoning
        # targets a converged model) and lets the attack/defense signal show up in
        # far fewer rounds than training from random init would need.
        payload = torch.load(init_checkpoint, map_location=device, weights_only=True)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        init_model.load_state_dict(state_dict)
    initial_state = _state_to_numpy(init_model.state_dict())

    results: list[dict[str, Any]] = []
    round_logs: list[RoundLog] = []

    for scenario in run_scenarios:
        # Index into the full module constant (not position within run_scenarios) so a
        # narrowed run reproduces the exact same run_seed - and therefore numerically
        # identical results - as the corresponding cell of a full run would use.
        scenario_idx = ATTACK_SCENARIOS.index(scenario)
        malicious = set(malicious_client_ids) if scenario != "none" else set()
        for mode in run_modes:
            mode_idx = AGGREGATION_MODES.index(mode)
            run_seed = seed + scenario_idx * 1000 + mode_idx * 10
            global_state = {k: v.copy() for k, v in initial_state.items()}
            detector = PCAClusterEMAFilter(
                pca_components=int(integ_cfg["pca_components"]),
                cluster_method=integ_cfg["cluster_method"],
                min_benign_fraction=float(integ_cfg["min_benign_fraction"]),
                ema_alpha=float(integ_cfg["ema_alpha"]),
                reject_below_trust=float(integ_cfg["reject_below_trust"]),
            )
            model = model_from_metadata(config).to(device)

            for round_index in range(num_rounds):
                client_states: dict[int, dict[str, np.ndarray]] = {}
                client_weights: dict[int, int] = {}
                for client_id in range(num_clients):
                    label_transform = None
                    if scenario == "label_flip" and client_id in malicious:
                        flip_seed = run_seed + round_index

                        def label_transform(y, _ni=normal_index, _sd=flip_seed):
                            return label_flip_to_normal(y, _ni, 1.0, _sd)

                    loader = _client_train_loader(
                        prepared, output_dir, client_id, seq_len, stride, batch_size,
                        max_samples_per_client, seed=run_seed + client_id * 100 + round_index,
                        label_transform=label_transform,
                    )
                    model.load_state_dict(_numpy_to_device_tensors(global_state, device))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                    train_one_epoch(model, loader, optimizer, criterion, device)
                    trained_state = _state_to_numpy(model.state_dict())

                    if client_id in malicious and scenario == "sign_flip":
                        trained_state = sign_flip(trained_state, scale=1.0)
                    elif client_id in malicious and scenario == "gaussian_noise":
                        trained_state = gaussian_noise(
                            trained_state, sigma=float(integ_cfg.get("attack_gaussian_sigma", 0.5)), seed=run_seed + round_index
                        )
                    elif client_id in malicious and scenario == "gradual_scaling":
                        trained_state = gradual_scaling(
                            trained_state, global_state, round_index,
                            growth_rate=float(integ_cfg.get("attack_gradual_scaling_growth", 1.0)),
                        )

                    client_states[client_id] = trained_state
                    client_weights[client_id] = len(loader.dataset)

                start = time.perf_counter()
                client_ids_sorted = sorted(client_states)
                caught_by: dict[str, str] = {}
                if mode == "no_defense":
                    accepted_ids = client_ids_sorted
                    new_state = weighted_average([client_states[c] for c in accepted_ids], [client_weights[c] for c in accepted_ids])
                elif mode == "coordinate_median":
                    accepted_ids = client_ids_sorted
                    new_state = coordinate_median([client_states[c] for c in accepted_ids])
                elif mode == "trimmed_mean":
                    accepted_ids = client_ids_sorted
                    new_state = trimmed_mean([client_states[c] for c in accepted_ids], trim_ratio=float(integ_cfg["trim_ratio"]))
                elif mode == "pca_cluster_ema":
                    accepted_ids = []
                    valid_ids = []
                    for c in client_ids_sorted:
                        validation = validate_update(
                            client_states[c], global_state,
                            max_abs_parameter=float(integ_cfg["max_abs_parameter"]),
                            max_update_norm_ratio=float(integ_cfg["max_update_norm_ratio"]),
                        )
                        if validation.accepted:
                            valid_ids.append(c)
                        else:
                            caught_by[str(c)] = "validation"
                    if valid_ids:
                        vectors = np.stack([flatten_delta(client_states[c], global_state) for c in valid_ids])
                        detection = detector.detect([str(c) for c in valid_ids], vectors)
                        for i, c in enumerate(valid_ids):
                            if detection.benign_mask[i]:
                                accepted_ids.append(c)
                            else:
                                caught_by[str(c)] = "detector"
                    new_state = (
                        weighted_average([client_states[c] for c in accepted_ids], [client_weights[c] for c in accepted_ids])
                        if accepted_ids
                        else global_state
                    )
                else:
                    raise ValueError(f"unknown aggregation mode: {mode}")
                aggregation_seconds = time.perf_counter() - start

                rejected_ids = sorted(set(client_ids_sorted) - set(accepted_ids))
                round_logs.append(
                    RoundLog(scenario, mode, round_index, sorted(malicious), rejected_ids, caught_by, aggregation_seconds)
                )
                global_state = new_state
                if on_progress:
                    on_progress(scenario, mode, round_index, num_rounds)

            model.load_state_dict(_numpy_to_device_tensors(global_state, device))
            test_metrics = evaluate_epoch(model, test_loader, criterion, device)
            results.append(
                {
                    "scenario": scenario,
                    "mode": mode,
                    "test_macro_f1": test_metrics.macro_f1,
                    "test_accuracy": test_metrics.accuracy,
                }
            )

    tp = fp = tn = fn = 0
    for log in round_logs:
        if log.mode != "pca_cluster_ema" or not log.malicious_ids:
            continue
        for client_id in range(num_clients):
            is_malicious = client_id in log.malicious_ids
            is_rejected = client_id in log.rejected_ids
            if is_malicious and is_rejected:
                tp += 1
            elif is_malicious and not is_rejected:
                fn += 1
            elif not is_malicious and is_rejected:
                fp += 1
            else:
                tn += 1

    detection_metrics = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / (tp + fn) if (tp + fn) else None,
        "fpr": fp / (fp + tn) if (fp + tn) else None,
        "tnr": tn / (tn + fp) if (tn + fp) else None,
    }

    avg_overhead = {}
    for mode in AGGREGATION_MODES:
        times = [log.aggregation_seconds for log in round_logs if log.mode == mode]
        avg_overhead[mode] = sum(times) / len(times) if times else None

    return {
        "malicious_client_ids": sorted(malicious_client_ids),
        "num_rounds": num_rounds,
        "max_samples_per_client": max_samples_per_client,
        "results": results,
        "detection_metrics": detection_metrics,
        "aggregation_overhead_seconds": avg_overhead,
        "round_logs": [
            {
                "scenario": log.scenario,
                "mode": log.mode,
                "round": log.round,
                "malicious_ids": log.malicious_ids,
                "rejected_ids": log.rejected_ids,
                "caught_by": log.caught_by,
                "aggregation_seconds": log.aggregation_seconds,
            }
            for log in round_logs
        ],
    }
