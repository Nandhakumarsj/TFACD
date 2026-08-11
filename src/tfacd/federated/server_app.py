from __future__ import annotations

from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx

from tfacd.common.config import load_config
from tfacd.federated.common import model_from_metadata
from tfacd.federated.integrity_strategy import IntegrityAwareStrategy
from tfacd.integrity.certification import write_manifest

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = load_config(str(context.run_config["config-path"]))
    model = model_from_metadata(config)
    initial_arrays = ArrayRecord(model.state_dict())

    # FedProx generalizes FedAvg: proximal-mu=0.0 is exactly FedAvg (Flower's own
    # strategy docs note this), so one strategy class covers both the Milestone 1
    # federation-correctness baseline and the non-IID heterogeneity comparison.
    proximal_mu = float(context.run_config.get("proximal-mu", 0.0))
    # Milestone 2: FTIL is wired into the live strategy now that Gate 4's
    # standalone benchmark has measured its detector's TPR/FPR under attack.
    # use-ftil defaults on; set to false to reproduce the pre-Gate-4 baseline.
    use_ftil = bool(context.run_config.get("use-ftil", True))
    integrity_cfg = config["integrity"]

    common_kwargs = dict(
        fraction_train=float(context.run_config["fraction-train"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
        proximal_mu=proximal_mu,
    )
    if use_ftil:
        strategy = IntegrityAwareStrategy(
            **common_kwargs,
            max_abs_parameter=float(integrity_cfg["max_abs_parameter"]),
            max_update_norm_ratio=float(integrity_cfg["max_update_norm_ratio"]),
            pca_components=int(integrity_cfg["pca_components"]),
            cluster_method=integrity_cfg["cluster_method"],
            min_benign_fraction=float(integrity_cfg["min_benign_fraction"]),
            ema_alpha=float(integrity_cfg["ema_alpha"]),
            reject_below_trust=float(integrity_cfg["reject_below_trust"]),
            aggregation_method=integrity_cfg.get("aggregation_method", "trimmed_mean"),
            trim_ratio=float(integrity_cfg["trim_ratio"]),
        )
    else:
        strategy = FedProx(**common_kwargs)

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": float(context.run_config["learning-rate"])}),
        num_rounds=int(context.run_config["num-server-rounds"]),
    )

    output_dir = Path("artifacts/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / ("flower_ftil_final.pt" if use_ftil else "flower_fedavg_final.pt")
    torch.save(result.arrays.to_torch_state_dict(), checkpoint)
    write_manifest(
        checkpoint,
        {
            "strategy": ("IntegrityAwareStrategy" if use_ftil else "FedAvg" if proximal_mu == 0.0 else "FedProx"),
            "proximal_mu": proximal_mu,
            "rounds": int(context.run_config["num-server-rounds"]),
            "status": "integrity-filtered" if use_ftil else "baseline-not-yet-integrity-filtered",
        },
    )
