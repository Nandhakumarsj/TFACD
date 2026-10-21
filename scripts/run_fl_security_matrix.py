from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader

from tfacd.common.config import load_config
from tfacd.common.reproducibility import seed_everything
from tfacd.data.dataset import SequenceDataset
from tfacd.data.preprocess import load_prepared
from tfacd.data.sequences import as_model_input
from tfacd.federated.common import model_from_metadata
from tfacd.integrity.aggregation import trimmed_mean, weighted_average
from tfacd.integrity.attacks import gaussian_noise, gradual_scaling, label_flip_to_normal, sign_flip, state_delta_l2_norm
from tfacd.integrity.detector import PCAClusterEMAFilter
from tfacd.integrity.update_validation import validate_update
from tfacd.integrity.vectorize import flatten_delta
from tfacd.training.engine import evaluate, predict_all, train_one_epoch

ATTACKS = ("label_flip", "sign_flip", "gaussian_noise", "gradual_scaling")
MODES = ("fedavg", "fedprox", "ftil")
FRACTIONS = (0.10, 0.20, 0.30)

parser = argparse.ArgumentParser(description="Run FedAvg/FedProx/FTIL under 0/10/20/30 percent malicious clients.")
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--partitions-dir", default=None)
parser.add_argument("--num-clients", type=int, default=10)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--proximal-mu", type=float, default=0.01)
parser.add_argument("--max-samples-per-client", type=int, default=15000)
parser.add_argument("--seed", type=int, default=None, help="override config seed for a reproducible replicate")
parser.add_argument("--init-checkpoint", default=None)
parser.add_argument("--attacks", nargs="+", choices=ATTACKS, default=list(ATTACKS))
parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
parser.add_argument("--output", default="artifacts/second_review/fl_security_matrix.json")
args = parser.parse_args()

if args.num_clients < 10:
    raise SystemExit("Use --num-clients 10 for exact 10%, 20%, and 30% malicious-client fractions.")

config = load_config(args.config)
config = json.loads(json.dumps(config))
if args.seed is not None:
    config["seed"] = int(args.seed)
config["federated"]["num_clients"] = args.num_clients
seed_everything(int(config.get("seed", 42)))
prepared = load_prepared(config["data"]["output_dir"])
metadata = json.loads((Path(config["data"]["output_dir"]) / "metadata.json").read_text(encoding="utf-8"))
classes = [str(c) for c in metadata["classes"]]
normal_index = classes.index("Normal") if "Normal" in classes else 0

partitions_dir = Path(args.partitions_dir or Path(config["data"]["output_dir"]) / "partitions")
missing = [i for i in range(args.num_clients) if not (partitions_dir / f"client_{i}.npy").exists()]
if missing:
    raise SystemExit(f"Missing partitions for clients {missing}. Generate a 10-client partition set first.")

seq_len = int(config["data"].get("sequence_length", 1))
stride = int(config["data"].get("sequence_stride", 1))
batch_size = int(config["training"]["batch_size"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_test, y_test = as_model_input(prepared.x_test, prepared.y_test, seq_len, stride)
test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()


def state_numpy(model):
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def state_tensors(state):
    return {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in state.items()}


def clone_state(state):
    return {key: np.asarray(value).copy() for key, value in state.items()}


def state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = np.ascontiguousarray(state[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


# One immutable initialization is shared by every mode/attack/fraction in this
# seed. A supplied checkpoint is authoritative; otherwise initialization is
# generated once after seeding, not independently for every experimental cell.
seed_everything(int(config.get("seed", 42)))
_initial_model = model_from_metadata(config).to(device)
if args.init_checkpoint:
    _payload = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
    _state = _payload["state_dict"] if isinstance(_payload, dict) and "state_dict" in _payload else _payload
    _initial_model.load_state_dict(_state)
INITIAL_STATE = state_numpy(_initial_model)
INITIAL_STATE_SHA256 = state_sha256(INITIAL_STATE)
TRAINING_SEED = int(config.get("seed", 42))


def client_loader(cid, seed):
    indices = np.load(partitions_dir / f"client_{cid}.npy")
    if args.max_samples_per_client and len(indices) > args.max_samples_per_client:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=args.max_samples_per_client, replace=False)
    x = prepared.x_train[indices]
    y = prepared.y_train[indices]
    x_seq, y_seq = as_model_input(x, y, seq_len, stride)
    return DataLoader(SequenceDataset(x_seq, y_seq), batch_size=batch_size, shuffle=True)


def evaluate_state(state):
    model = model_from_metadata(config).to(device)
    model.load_state_dict(state_tensors(state))
    metrics = evaluate(model, test_loader, criterion, device)
    labels, preds = predict_all(model, test_loader, device)
    attack_true = np.isin(labels, [i for i, c in enumerate(classes) if c != "Normal"])
    attack_pred = np.isin(preds, [i for i, c in enumerate(classes) if c != "Normal"])
    attack_recall = float(recall_score(attack_true, attack_pred, zero_division=0))
    attack_precision = float(precision_score(attack_true, attack_pred, zero_division=0))
    return {
        "accuracy": float(metrics.accuracy),
        "macro_precision": float(metrics.macro_precision),
        "macro_recall": float(metrics.macro_recall),
        "macro_f1": float(metrics.macro_f1),
        "attack_detection_rate": attack_recall,
        "attack_detection_precision": attack_precision,
    }


def run_one(mode, attack, malicious_fraction):
    # Identical client sampling/batch-order seed across paired methods. Attack
    # randomness and malicious-client selection use a separate derived seed.
    seed_everything(TRAINING_SEED)
    attack_seed = TRAINING_SEED + ATTACKS.index(attack) * 1000 + int(malicious_fraction * 100)
    rng = np.random.default_rng(attack_seed)
    global_state = clone_state(INITIAL_STATE)
    integ = config["integrity"]
    detector = PCAClusterEMAFilter(
        pca_components=int(integ["pca_components"]),
        cluster_method=integ["cluster_method"],
        min_benign_fraction=float(integ["min_benign_fraction"]),
        ema_alpha=float(integ["ema_alpha"]),
        reject_below_trust=float(integ["reject_below_trust"]),
    )
    malicious_count = int(round(args.num_clients * malicious_fraction))
    if malicious_count == 0:
        malicious_count = 1
    malicious_ids = set(rng.choice(args.num_clients, size=malicious_count, replace=False).tolist())
    round_detection = []
    attack_diagnostics = []

    for round_index in range(args.rounds):
        states = {}
        weights = {}
        for cid in range(args.num_clients):
            loader = client_loader(cid, TRAINING_SEED + cid * 100 + round_index)
            model = model_from_metadata(config).to(device)
            model.load_state_dict(state_tensors(global_state))
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
            global_params = [state_tensors(global_state)[name] for name, _ in model.named_parameters()]
            label_transform = None
            if attack == "label_flip" and cid in malicious_ids:
                flip_seed = attack_seed + round_index
                label_transform = lambda y, _ni=normal_index, _sd=flip_seed + cid * 17: label_flip_to_normal(y, _ni, 1.0, _sd)
                # label transform has to be applied to the actual dataset labels.
                indices = np.load(partitions_dir / f"client_{cid}.npy")
                if args.max_samples_per_client and len(indices) > args.max_samples_per_client:
                    indices = np.random.default_rng(TRAINING_SEED + cid * 100 + round_index).choice(indices, size=args.max_samples_per_client, replace=False)
                y = label_transform(prepared.y_train[indices])
                x = prepared.x_train[indices]
                x_seq, y_seq = as_model_input(x, y, seq_len, stride)
                loader = DataLoader(SequenceDataset(x_seq, y_seq), batch_size=batch_size, shuffle=True)
            train_one_epoch(
                model, loader, optimizer, criterion, device,
                proximal_mu=args.proximal_mu if mode == "fedprox" else 0.0,
                global_params=global_params,
            )
            trained = state_numpy(model)

            if cid in malicious_ids:
                if attack == "sign_flip":
                    trained = sign_flip(
                        trained,
                        global_state,
                        scale=1.0,
                    )
                elif attack == "gaussian_noise":
                    update_norm = state_delta_l2_norm(trained, global_state)
                    poisoned = gaussian_noise(
                        trained, global_state,
                        sigma=float(integ.get("attack_gaussian_sigma", 0.5)),
                        seed=attack_seed + round_index * 100 + cid,
                    )
                    noise_norm = state_delta_l2_norm(poisoned, trained)
                    attack_diagnostics.append({
                        "client_id": cid, "round": round_index + 1,
                        "clean_update_l2_norm": update_norm,
                        "injected_noise_l2_norm": noise_norm,
                        "observed_noise_to_update_norm_ratio": noise_norm / update_norm if update_norm > 0 else None,
                        "configured_relative_sigma": float(integ.get("attack_gaussian_sigma", 0.5)),
                    })
                    trained = poisoned
                elif attack == "gradual_scaling":
                    trained = gradual_scaling(trained, global_state, round_index, growth_rate=float(integ.get("attack_gradual_scaling_growth", 1.0)))
            states[cid] = trained
            weights[cid] = len(loader.dataset)

        ids = sorted(states)
        caught = set()
        if mode == "ftil":
            valid = []
            for cid in ids:
                result = validate_update(states[cid], global_state, max_abs_parameter=float(integ["max_abs_parameter"]), max_update_norm_ratio=float(integ["max_update_norm_ratio"]))
                if result.accepted:
                    valid.append(cid)
                else:
                    caught.add(cid)
            accepted = []
            if valid:
                vectors = np.stack([flatten_delta(states[cid], global_state) for cid in valid])
                detection = detector.detect([str(cid) for cid in valid], vectors)
                accepted = [cid for cid, keep in zip(valid, detection.benign_mask) if keep]
                caught.update(cid for cid, keep in zip(valid, detection.benign_mask) if not keep)
            if accepted:
                global_state = trimmed_mean([states[cid] for cid in accepted], trim_ratio=float(integ["trim_ratio"]))
            else:
                global_state = {k: v.copy() for k, v in global_state.items()}
        else:
            global_state = weighted_average([states[cid] for cid in ids], [weights[cid] for cid in ids])

        if mode == "ftil":
            tp = sum(cid in malicious_ids and cid in caught for cid in ids)
            fn = sum(cid in malicious_ids and cid not in caught for cid in ids)
            fp = sum(cid not in malicious_ids and cid in caught for cid in ids)
            tn = sum(cid not in malicious_ids and cid not in caught for cid in ids)
            round_detection.append({"tp": tp, "fn": fn, "fp": fp, "tn": tn})

    metrics = evaluate_state(global_state)
    return metrics, malicious_ids, round_detection, attack_diagnostics


# Clean baselines use the same initialization, training seed, partitions, and
# per-client minibatch order as the paired attacked conditions. Previous files in
# artifacts/integrity are preserved; this run writes to the second-review path.
clean_by_mode = {}
clean_client_confusion = {}
results = []
for mode in args.modes:
    seed_everything(TRAINING_SEED)
    global_state = clone_state(INITIAL_STATE)
    clean_detector = None
    confusion = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    if mode == "ftil":
        clean_detector = PCAClusterEMAFilter(
            pca_components=int(config["integrity"]["pca_components"]),
            cluster_method=config["integrity"]["cluster_method"],
            min_benign_fraction=float(config["integrity"]["min_benign_fraction"]),
            ema_alpha=float(config["integrity"]["ema_alpha"]),
            reject_below_trust=float(config["integrity"]["reject_below_trust"]),
        )
    for round_index in range(args.rounds):
        states, weights = {}, {}
        for cid in range(args.num_clients):
            loader = client_loader(cid, TRAINING_SEED + cid * 100 + round_index)
            model = model_from_metadata(config).to(device)
            model.load_state_dict(state_tensors(global_state))
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
            global_params = [state_tensors(global_state)[name] for name, _ in model.named_parameters()]
            train_one_epoch(
                model, loader, optimizer, criterion, device,
                proximal_mu=args.proximal_mu if mode == "fedprox" else 0.0,
                global_params=global_params,
            )
            states[cid] = state_numpy(model)
            weights[cid] = len(loader.dataset)
        if mode == "ftil":
            valid, caught = [], set()
            for cid in range(args.num_clients):
                validation = validate_update(
                    states[cid], global_state,
                    max_abs_parameter=float(config["integrity"]["max_abs_parameter"]),
                    max_update_norm_ratio=float(config["integrity"]["max_update_norm_ratio"]),
                )
                if validation.accepted:
                    valid.append(cid)
                else:
                    caught.add(cid)
            if valid:
                vectors = np.stack([flatten_delta(states[cid], global_state) for cid in valid])
                det = clean_detector.detect([str(c) for c in valid], vectors)
                accepted = [cid for cid, keep in zip(valid, det.benign_mask) if keep]
                caught.update(cid for cid, keep in zip(valid, det.benign_mask) if not keep)
            else:
                accepted = []
            for cid in range(args.num_clients):
                if cid in caught:
                    confusion["fp"] += 1
                else:
                    confusion["tn"] += 1
            global_state = (
                trimmed_mean([states[c] for c in accepted], trim_ratio=float(config["integrity"]["trim_ratio"]))
                if accepted else clone_state(global_state)
            )
        else:
            global_state = weighted_average(
                [states[c] for c in range(args.num_clients)],
                [weights[c] for c in range(args.num_clients)],
            )
    clean_by_mode[mode] = evaluate_state(global_state)
    clean_client_confusion[mode] = confusion if mode == "ftil" else None

# Export the explicit 0%-malicious condition; it is not inferred only from a
# separate baseline field. The same baseline doubles as the degradation reference.
for mode in args.modes:
    cc = clean_client_confusion[mode]
    fpr = cc["fp"] / (cc["fp"] + cc["tn"]) if cc and (cc["fp"] + cc["tn"]) else (0.0 if cc else None)
    results.append({
        "mode": mode, "attack": "none", "malicious_fraction_requested": 0.0,
        "malicious_clients": [], "effective_malicious_fraction": 0.0,
        "metrics": clean_by_mode[mode], "malicious_client_detection_rate": None,
        "benign_client_false_positive_rate": fpr,
        "client_level_confusion": cc, "attack_diagnostics": [],
        "global_model_degradation_macro_f1": 0.0, "global_model_degradation_accuracy": 0.0,
        "runtime_seconds": 0.0, "training_seed": TRAINING_SEED,
    })

for attack in args.attacks:
    for fraction in FRACTIONS:
        for mode in args.modes:
            started = time.perf_counter()
            metrics, malicious_ids, detections, attack_diagnostics = run_one(mode, attack, fraction)
            detection_rate = None
            client_confusion = None
            benign_fpr = None
            if detections:
                client_confusion = {key: sum(int(x[key]) for x in detections) for key in ("tp", "fn", "fp", "tn")}
                tp, fn = client_confusion["tp"], client_confusion["fn"]
                detection_rate = tp / (tp + fn) if (tp + fn) else None
                fp, tn = client_confusion["fp"], client_confusion["tn"]
                benign_fpr = fp / (fp + tn) if (fp + tn) else None
            clean = clean_by_mode[mode]
            results.append({
                "mode": mode,
                "attack": attack,
                "malicious_fraction_requested": fraction,
                "malicious_clients": sorted(malicious_ids),
                "effective_malicious_fraction": len(malicious_ids) / args.num_clients,
                "metrics": metrics,
                "malicious_client_detection_rate": detection_rate,
                "benign_client_false_positive_rate": benign_fpr,
                "client_level_confusion": client_confusion,
                "attack_diagnostics": attack_diagnostics,
                "global_model_degradation_macro_f1": clean["macro_f1"] - metrics["macro_f1"],
                "global_model_degradation_accuracy": clean["accuracy"] - metrics["accuracy"],
                "runtime_seconds": time.perf_counter() - started,
                "training_seed": TRAINING_SEED,
                "attack_seed": TRAINING_SEED + ATTACKS.index(attack) * 1000 + int(fraction * 100),
            })

out = {
    "experiment": "FTIL/FedAvg/FedProx client-poisoning security matrix",
    "seed": TRAINING_SEED,
    "initial_state_sha256": INITIAL_STATE_SHA256,
    "initialization_source": str(args.init_checkpoint) if args.init_checkpoint else "seeded_random_initialization",
    "command_argv": list(sys.argv),
    "config_path": str(args.config),
    "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
    "prepared_npz_sha256": hashlib.sha256((Path(config["data"]["output_dir"]) / "prepared.npz").read_bytes()).hexdigest(),
    "metadata_sha256": hashlib.sha256((Path(config["data"]["output_dir"]) / "metadata.json").read_bytes()).hexdigest(),
    "partition_manifest_sha256": hashlib.sha256((partitions_dir / "manifest.json").read_bytes()).hexdigest() if (partitions_dir / "manifest.json").exists() else None,
    "partition_files_sha256": {
        str(cid): hashlib.sha256((partitions_dir / f"client_{cid}.npy").read_bytes()).hexdigest()
        for cid in range(args.num_clients)
    },
    "init_checkpoint_sha256": hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest() if args.init_checkpoint else None,
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "torch_version": torch.__version__,
    "num_clients": args.num_clients,
    "rounds": args.rounds,
    "proximal_mu": args.proximal_mu,
    "attacks": args.attacks,
    "modes": args.modes,
    "fractions": [0.0, *FRACTIONS],
    "clean_baselines": clean_by_mode,
    "clean_client_detection": clean_client_confusion,
    "client_metric_definitions": {
        "malicious_client_detection_rate": "TP / (TP + FN), FTIL only",
        "benign_client_false_positive_rate": "FP / (FP + TN), FTIL only",
        "client_level_confusion": "aggregated client-round TP/FN/FP/TN; distinct from record-level IDS attack metrics",
    },
    "gaussian_noise_semantics": {
        "definition": "global-relative client update plus Gaussian noise; sigma is a relative L2/update-RMS strength",
        "configured_sigma": float(config["integrity"].get("attack_gaussian_sigma", 0.5)),
        "diagnostics": "per malicious client-round update norm, injected noise norm, and observed ratio are stored with each result",
    },
    "results": results,
}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print(f"Saved: {path.resolve()}")
