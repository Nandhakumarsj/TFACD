import argparse

from tfacd.common.config import load_config
from tfacd.data.preprocess import preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
args = parser.parse_args()
result = preprocess(load_config(args.config))
print(f"Prepared feature_dim={result.feature_dim}, num_classes={result.num_classes}")
