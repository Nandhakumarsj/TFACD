from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
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
from tfacd.integrity.attacks import gaussian_noise, gradual_scaling, label_flip_to_normal, sign_flip
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
parser.add_argument("--init-checkpoint", default=None)
parser.add_argument("--attacks", nargs="+", choices=ATTACKS, default=list(ATTACKS))
parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
parser.add_argument("--output", default="artifacts/integrity/fl_security_matrix.json")
args = parser.parse_args()

if args.num_clients < 10:
    raise SystemExit("Use --num-clients 10 for exact 10%, 20%, and 30% malicious-client fractions.")

config = load_config(args.config)
config = json.loads(json.dumps(config))
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
    seed = int(config.get("seed", 42)) + ATTACKS.index(attack) * 1000 + int(malicious_fraction * 100)
    rng = np.random.default_rng(seed)
    base_model = model_from_metadata(config).to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        base_model.load_state_dict(state)
    initial_state = state_numpy(base_model)
    global_state = {k: v.copy() for k, v in initial_state.items()}
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

    for round_index in range(args.rounds):
        states = {}
        weights = {}
        for cid in range(args.num_clients):
            loader = client_loader(cid, seed + cid * 100 + round_index)
            model = model_from_metadata(config).to(device)
            model.load_state_dict(state_tensors(global_state))
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
            global_params = [state_tensors(global_state)[name] for name, _ in model.named_parameters()]
            label_transform = None
            if attack == "label_flip" and cid in malicious_ids:
                flip_seed = seed + round_index
                label_transform = lambda y, _ni=normal_index, _sd=flip_seed + cid * 17: label_flip_to_normal(y, _ni, 1.0, _sd)
                # label transform has to be applied to the actual dataset labels.
                indices = np.load(partitions_dir / f"client_{cid}.npy")
                if args.max_samples_per_client and len(indices) > args.max_samples_per_client:
                    indices = np.random.default_rng(seed + cid * 100 + round_index).choice(indices, size=args.max_samples_per_client, replace=False)
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
                    trained = gaussian_noise(trained, sigma=float(integ.get("attack_gaussian_sigma", 0.5)), seed=seed + round_index * 100 + cid)
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
    clean_metrics = None
    return metrics, malicious_ids, round_detection


# Clean results are run once per mode and used as the degradation reference.
clean_by_mode = {}
results = []
for mode in args.modes:
    mode_seed = int(config.get("seed", 42)) + MODES.index(mode) * 10000
    base_model = model_from_metadata(config).to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        base_model.load_state_dict(state)
    global_state = state_numpy(base_model)
    clean_detector = None
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
            loader = client_loader(cid, mode_seed + cid * 100 + round_index)
            model = model_from_metadata(config).to(device)
            model.load_state_dict(state_tensors(global_state))
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
            global_params = [state_tensors(global_state)[name] for name, _ in model.named_parameters()]
            train_one_epoch(model, loader, optimizer, criterion, device, proximal_mu=args.proximal_mu if mode == "fedprox" else 0.0, global_params=global_params)
            states[cid] = state_numpy(model)
            weights[cid] = len(loader.dataset)
        if mode == "ftil":
            valid = []
            for cid in range(args.num_clients):
                if validate_update(states[cid], global_state, max_abs_parameter=float(config["integrity"]["max_abs_parameter"]), max_update_norm_ratio=float(config["integrity"]["max_update_norm_ratio"])).accepted:
                    valid.append(cid)
            if valid:
                vectors = np.stack([flatten_delta(states[cid], global_state) for cid in valid])
                det = clean_detector.detect([str(c) for c in valid], vectors)
                accepted = [cid for cid, keep in zip(valid, det.benign_mask) if keep]
            else:
                accepted = []
            global_state = trimmed_mean([states[c] for c in accepted], trim_ratio=float(config["integrity"]["trim_ratio"])) if accepted else global_state
        else:
            global_state = weighted_average([states[c] for c in range(args.num_clients)], [weights[c] for c in range(args.num_clients)])
    clean_by_mode[mode] = evaluate_state(global_state)

for attack in args.attacks:
    for fraction in FRACTIONS:
        for mode in args.modes:
            started = time.perf_counter()
            metrics, malicious_ids, detections = run_one(mode, attack, fraction)
            detection_rate = None
            if detections:
                tp = sum(x["tp"] for x in detections)
                fn = sum(x["fn"] for x in detections)
                detection_rate = tp / (tp + fn) if (tp + fn) else None
            clean = clean_by_mode[mode]
            results.append({
                "mode": mode,
                "attack": attack,
                "malicious_fraction_requested": fraction,
                "malicious_clients": sorted(malicious_ids),
                "effective_malicious_fraction": len(malicious_ids) / args.num_clients,
                "metrics": metrics,
                "malicious_client_detection_rate": detection_rate,
                "global_model_degradation_macro_f1": clean["macro_f1"] - metrics["macro_f1"],
                "global_model_degradation_accuracy": clean["accuracy"] - metrics["accuracy"],
                "runtime_seconds": time.perf_counter() - started,
            })

out = {
    "num_clients": args.num_clients,
    "rounds": args.rounds,
    "proximal_mu": args.proximal_mu,
    "attacks": args.attacks,
    "modes": args.modes,
    "fractions": list(FRACTIONS),
    "clean_baselines": clean_by_mode,
    "results": results,
}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print(f"Saved: {path.resolve()}")
