# TFACD Phase-I Starter

**Trustworthy Federated Agentic Cyber Defense for IIoT**

This repository starts with the **training plane** and leaves stable interfaces for the runtime plane.

## Development order

1. Inspect the local Edge-IIoTset CSV and freeze its schema.
2. Build a leakage-safe centralized baseline.
3. Create IID and non-IID client partitions.
4. Run Flower FedAvg/FedProx simulations.
5. Add the Federated Trust & Integrity Layer (FTIL):
   - structural update validation;
   - FedDMC-inspired PCA + clustering baseline;
   - historical EMA client trust;
   - robust aggregation comparisons;
   - hash/signature-based certified model release.
6. Only then connect the certified model to the runtime plane.

## Important scientific constraint

A CNN-BiLSTM only has a meaningful temporal interpretation when row order, timestamps, or flow/session grouping are preserved. The selected DNN CSV may be shuffled. Run the inspector first. If no defensible ordering exists, use `sequence_length: 1` as the baseline and do not claim temporal learning until packet/flow ordering is reconstructed.

## Windows + Quadro P5000

The Quadro P5000 is a Pascal GPU. CUDA 13 removed Pascal library/offline compilation support, so install a **PyTorch CUDA 12.6 wheel**, even though `nvidia-smi` reports driver CUDA 13 capability.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install PyTorch from the cu126 index selected from pytorch.org
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev,flower]"
python scripts/check_environment.py
```

`nvidia-smi` reports the maximum CUDA version supported by the driver; it does not require your Python package to use CUDA 13.

## Put the dataset outside Git

Edit `configs/edge_iiot.yaml`:

```yaml
data:
  raw_csv: "./datasets/Edge-IIoT/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
```

Then:

```powershell
python scripts/inspect_dataset.py --config configs/edge_iiot.yaml
python scripts/preprocess_dataset.py --config configs/edge_iiot.yaml
python scripts/train_centralized.py --config configs/edge_iiot.yaml
python scripts/create_partitions.py --config configs/edge_iiot.yaml
```

After the centralized baseline succeeds:

```powershell
flwr run . --stream
```

## Deployment-mode TLS (optional, not the default path)

Every command above uses Flower's local-simulation engine (single process, no real network sockets) - there is nothing to encrypt there. `scripts/run_deployment_smoke_test.ps1` is a separate, real verification path: it starts an actual `flower-superlink` + two `flower-supernode` processes as distinct OS processes talking over real TLS-encrypted sockets on `127.0.0.1`, and submits a live training round against them.

This is **not** "mTLS" - `flower-supernode --root-certificates`'s own docstring says "This is NOT a client certificate for mTLS." What's actually running is server-authenticated TLS (a local CA + SuperLink server certificate, `src/tfacd/security/certificates.py`) plus a separate SuperNode public-key node-authentication mechanism (EC/OpenSSH keypairs - **not** this repo's Ed25519 model-signing format, Flower requires a different key shape for node auth). Simulation stays the default day-to-day path; this is an alternate, tested one.

```powershell
python scripts/generate_deployment_certs.py    # CA + server cert, SuperNode auth keypairs -> artifacts/certs/
./scripts/run_deployment_smoke_test.ps1         # starts the processes, registers nodes, submits a real run, tears everything down
```

Manual sequence, if reproducing by hand: generate certs -> start `flower-superlink --ssl-certfile ... --ssl-keyfile ... --ssl-ca-certfile ... --enable-supernode-auth` -> `flwr supernode register <pubkey> local-tls` per node (returns a `node_id`, requires the SuperLink already running) -> start each `flower-supernode --auth-supernode-private-key ... --root-certificates ... --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:909<N>` (each SuperNode needs a distinct ClientAppIo port when running more than one on the same machine) -> `flwr run . local-tls --stream` (the SuperLink connection is a *positional* argument, not `--federation` - that flag is for Flower's hosted/cloud federation IDs in `@account/name` form, unrelated to selecting a local connection).

## What is implemented now

- dataset schema inspection and leakage warnings;
- mixed numeric/categorical preprocessing;
- train-only fitting of encoders/scalers;
- CNN-BiLSTM model for `[batch, sequence, features]` input;
- centralized training/evaluation;
- Dirichlet non-IID client partitioning;
- Flower `ClientApp`/`ServerApp` starter;
- structural update validation;
- FedDMC-inspired PCA/clustering/EMA detector (explicitly not an exact reproduction);
- weighted mean, coordinate median, and trimmed-mean aggregation utilities;
- model hashing and optional Ed25519 signing;
- tests for the model and integrity utilities;
- runtime-plane package interfaces reserved for the next milestone.

## Initial experiment matrix

| Experiment | Purpose |
|---|---|
| Centralized MLP | sanity baseline |
| Centralized CNN-BiLSTM, sequence=1 | architecture baseline without temporal claim |
| Centralized CNN-BiLSTM, sequence>1 | only after ordering audit |
| Flower FedAvg IID | federation correctness |
| Flower FedAvg Dirichlet non-IID | heterogeneity baseline |
| Flower FedProx non-IID | heterogeneity comparison |
| FedAvg under label-flip clients | poisoning baseline |
| PCA-cluster-EMA filtering + FedAvg | FTIL baseline |
| Coordinate median / trimmed mean | robust aggregation baselines |

## Repository map

```text
tfacd_phase1_starter/
├── configs/
├── docs/
├── scripts/
├── src/tfacd/
│   ├── common/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── federated/
│   ├── integrity/
│   └── runtime/
├── tests/
├── artifacts/             # generated, ignored
├── pyproject.toml
└── README.md
```

## Research honesty

- `PCAClusterEMAFilter` is **inspired by** FedDMC's dimensionality reduction, clustering, and historical correction. It is not the paper's exact BTBCN implementation.
- Cross-round FLTracer-like features are a later milestone after the baseline round history is stable.
- Do not claim “Byzantine robust” unless at least one recognized robust strategy is benchmarked under defined malicious-client ratios.
- TLS is tested after the local simulation works; it is a deployment control, not a replacement for malicious-update detection.
