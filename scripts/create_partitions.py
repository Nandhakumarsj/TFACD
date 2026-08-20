import argparse
from pathlib import Path

from tfacd.common.config import load_config
from tfacd.data.partition import dirichlet_partition, iid_partition, save_partitions
from tfacd.data.preprocess import load_prepared

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
args = parser.parse_args()
config = load_config(args.config)
prepared = load_prepared(config["data"]["output_dir"])
fed = config["federated"]
if fed["partition_mode"] == "iid":
    partitions = iid_partition(prepared.y_train, int(fed["num_clients"]), int(config["seed"]))
else:
    partitions = dirichlet_partition(
        prepared.y_train,
        int(fed["num_clients"]),
        float(fed["dirichlet_alpha"]),
        int(config["seed"]),
        int(fed["min_samples_per_client"]),
        min_class_samples=int(fed.get("min_class_samples_per_client", 0)),
    )
out = Path(config["data"]["output_dir"]) / "partitions"
save_partitions(partitions, out, prepared.y_train)
print(f"Saved {len(partitions)} partitions to {out}")
