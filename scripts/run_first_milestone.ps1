param(
  [string]$Config = "configs/edge_iiot.yaml"
)
$ErrorActionPreference = "Stop"
python scripts/check_environment.py
python scripts/inspect_dataset.py --config $Config
python scripts/preprocess_dataset.py --config $Config
python scripts/train_centralized.py --config $Config
python scripts/create_partitions.py --config $Config
Write-Host "Centralized baseline and partitions complete. Review metrics before running Flower."
