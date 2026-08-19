import argparse

from tfacd.common.config import load_config
from tfacd.common.reproducibility import seed_everything
from tfacd.training.centralized import run_centralized

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
args = parser.parse_args()
config = load_config(args.config)
seed_everything(int(config.get("seed", 42)))
print(run_centralized(config))
