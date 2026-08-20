from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from tfacd.common.config import load_config
from tfacd.federated.common import device, model_from_metadata
from tfacd.federated.loaders import client_loaders
from tfacd.training.engine import class_weights, evaluate, train_one_epoch

app = ClientApp()


def _weighted_criterion(config: dict, trainloader, run_device, num_classes: int) -> torch.nn.Module:
    # Per-client LOCAL class weighting - configs/edge_iiot.yaml's training.class_weighting
    # was applied in the centralized baseline (training/centralized.py:50-51) but was
    # silently dropped here (unweighted CrossEntropyLoss(), config never consulted).
    # Weighted from THIS client's own local partition, not the global distribution -
    # Dirichlet non-IID partitioning means a client's local class balance can differ
    # sharply from the dataset-wide one (see partition.py's dirichlet_partition).
    if not config["training"].get("class_weighting", True):
        return torch.nn.CrossEntropyLoss()
    weights = class_weights(trainloader.dataset.y.numpy(), num_classes).to(run_device)
    return torch.nn.CrossEntropyLoss(weight=weights)


@app.train()
def train(msg: Message, context: Context) -> Message:
    config = load_config(str(context.run_config["config-path"]))
    model = model_from_metadata(config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    run_device = device()
    model.to(run_device)
    proximal_mu = float(msg.content["config"].get("proximal-mu", 0.0))
    global_params = [p.detach().clone() for p in model.parameters()] if proximal_mu > 0.0 else None
    batch_size = int(context.run_config["batch-size"])
    partition_id = int(context.node_config["partition-id"])
    trainloader, _ = client_loaders(config, partition_id, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(msg.content["config"]["lr"]))
    num_classes = model.classifier[-1].out_features
    criterion = _weighted_criterion(config, trainloader, run_device, num_classes)
    metrics = None
    for _ in range(int(context.run_config["local-epochs"])):
        metrics = train_one_epoch(
            model, trainloader, optimizer, criterion, run_device,
            proximal_mu=proximal_mu, global_params=global_params,
        )
    assert metrics is not None
    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "train_loss": float(metrics.loss),
                    "train_macro_f1": float(metrics.macro_f1),
                    "num-examples": len(trainloader.dataset),
                }
            ),
            "client-metadata": ConfigRecord({"client-id": str(partition_id)}),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context) -> Message:
    config = load_config(str(context.run_config["config-path"]))
    model = model_from_metadata(config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    run_device = device()
    model.to(run_device)
    partition_id = int(context.node_config["partition-id"])
    trainloader, valloader = client_loaders(config, partition_id, int(context.run_config["batch-size"]))
    # Same weighted criterion as train() (weights derived from the train split, matching
    # training/centralized.py's pattern) - only the reported eval_loss scale is affected,
    # not eval_acc/eval_macro_f1 (those come from predictions vs. labels directly).
    num_classes = model.classifier[-1].out_features
    criterion = _weighted_criterion(config, trainloader, run_device, num_classes)
    metrics = evaluate(model, valloader, criterion, run_device)
    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_loss": float(metrics.loss),
                        "eval_acc": float(metrics.accuracy),
                        "eval_macro_f1": float(metrics.macro_f1),
                        "num-examples": len(valloader.dataset),
                    }
                )
            }
        ),
        reply_to=msg,
    )
