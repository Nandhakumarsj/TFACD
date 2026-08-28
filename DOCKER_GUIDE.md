# TFACD – Docker & Containerised Execution Guide

> **Platform note**: The project targets a **CUDA 12.6 / Quadro P5000** workstation.  
> The Docker image installs the matching PyTorch CUDA 12.6 wheel automatically.  
> A CPU-only run is possible but will be 10–50× slower for FL simulation.

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|-----------------|-------|
| Docker Engine | 25.0 | `docker --version` |
| Docker Compose (plugin) | 2.20 | `docker compose version` |
| NVIDIA Container Toolkit | latest | Required for GPU pass-through |
| Ollama | 0.3 | Must be running on the **host** at `localhost:11434` |

Install NVIDIA Container Toolkit (Ubuntu/WSL2):

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Quick Start

### 1 – Place the dataset

The raw CSV is never copied into the image; it is mounted read-only:

```
datasets/
└── Edge_IIoT/
    └── Selected dataset for ML and DL/
        └── DNN-EdgeIIoT-dataset.csv   ← 2.2 GB
```

This path is already configured in `configs/edge_iiot.yaml → data.raw_csv`.

### 2 – Build images

```bash
docker compose build training

docker compose build runtime
```

### 3 – Run the full training pipeline

Each command below maps to a sequential gate from the scientific roadmap.

```bash
# --- Gate 0: Dataset inspection ---
docker compose run --rm training \
  python scripts/inspect_dataset.py --config configs/edge_iiot.yaml

# --- Gate 1: Centralized CNN-BiLSTM baseline ---
docker compose run --rm training \
  python scripts/train_centralized.py --config configs/edge_iiot.yaml

# Optional: visualise training curves (opens matplotlib window on host display)
docker compose run --rm -e DISPLAY=$DISPLAY \
  --volume /tmp/.X11-unix:/tmp/.X11-unix training \
  python scripts/visualize_centralized.py

# --- Gate 2: Create Dirichlet partitions for FL ---
docker compose run --rm training \
  python scripts/create_partitions.py --config configs/edge_iiot.yaml

# --- Gate 3: Flower FL simulation (FedAvg / FedProx) + FTIL ---
docker compose run --rm flower

# --- Gate 4: FTIL integrity & attack benchmark ---
docker compose run --rm integrity

# --- Gate 5: Certify global model ---
docker compose run --rm training \
  python scripts/certify_model.py --config configs/edge_iiot.yaml

# Verify the certificate
docker compose run --rm training \
  python scripts/verify_certified_model.py --config configs/edge_iiot.yaml
```

### 4 – Run the runtime plane

> The Ollama service must already be running on the **host** before this step.

```bash
# Streaming demo (CSV replay against certified model)
docker compose run --rm runtime

# Trust boundary demo
docker compose run --rm runtime \
  python scripts/run_trust_boundary_demo.py --config configs/edge_iiot.yaml

# LLM engine benchmark (required before switching engine: "llm" in config)
docker compose run --rm runtime \
  python scripts/run_llm_engine_benchmark.py --config configs/edge_iiot.yaml

# Agentic LLM attack scenario
docker compose run --rm runtime \
  python scripts/run_attack_scenario.py --config configs/edge_iiot.yaml
```

---

## Image Targets

| Target | Purpose | Default CMD |
|--------|---------|-------------|
| `training` | Full training plane: baseline, FL, FTIL, certification | `train_centralized.py` |
| `runtime` | Runtime plane: streaming, agentic engine, trust boundary | `run_streaming_demo.py` |

Both targets share the same `builder` layer (cached pip install), so rebuilding
one does not reinstall packages for the other.

---

## Volume & Port Layout

```
./datasets  → /app/datasets  (ro)  Raw Edge-IIoTset CSV
./artifacts → /app/artifacts       Models, keys, audit logs, benchmark reports
./configs   → built into image     Edit on host and rebuild, or mount as volume
```

Port `11434` is exposed by the `runtime` service for Ollama routing.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU to expose to the container |
| `DISPLAY` | (unset) | Set to `$DISPLAY` to forward matplotlib GUI |
| `OLLAMA_HOST` | `http://host-gateway:11434` | Override if Ollama runs on a remote host |

---

## Running Without Docker (Native Windows / WSL2)

```powershell
# One-time setup (Windows / Quadro P5000)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev,flower,agentic-llm,viz,explainability,mqtt]"

# Verify
python scripts/check_environment.py
```

Then run any script directly:

```powershell
python scripts/train_centralized.py --config configs/edge_iiot.yaml
```

---

## Dependency Groups

| Group flag | Packages installed | When you need it |
|------------|--------------------|-----------------|
| *(core)* | numpy, pandas, scikit-learn, PyYAML, joblib, pydantic, rich, cryptography, sentence-transformers, torch, torchvision, torchaudio | Always |
| `[dev]` | pytest, ruff, mypy | Running tests / linting |
| `[flower]` | flwr[simulation] | Flower FL simulation |
| `[agentic-llm]` | langgraph, langchain-core, langchain-ollama | LLM decision engine |
| `[viz]` | matplotlib, seaborn | Training curve plots, confusion matrix |
| `[explainability]` | shap, lime | SHAP / LIME attribution reports |
| `[mqtt]` | paho-mqtt | MQTT live streaming source |

Install all groups at once (as the Docker builder stage does):

```bash
pip install -e ".[dev,flower,agentic-llm,viz,explainability,mqtt]"
```

---

## Running Tests

```bash
# Native
python -m pytest tests/ -v

# Inside a container
docker compose run --rm training python -m pytest tests/ -v
```

276 tests should pass, 1 skipped (Ollama integration – requires live Ollama).

---

## Troubleshooting

### `torch` CUDA not detected inside container

```bash
docker compose run --rm training python -c \
  "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If `False`, ensure:
1. The NVIDIA Container Toolkit is installed and Docker was restarted.
2. Your driver supports CUDA ≥ 12.6 (`nvidia-smi` shows `CUDA Version: 12.x`).

### `OllamaError` / `Connection refused` at port 11434

Ollama runs on the **host**, not inside the container.  
The compose file uses `host-gateway` networking automatically.  
If your Docker network doesn't resolve `host.docker.internal`, add:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

to the relevant service in `docker-compose.yml`.

### Out-of-memory during FL simulation

Reduce `num_clients` in `configs/edge_iiot.yaml → federated.num_clients` or
set `CUDA_VISIBLE_DEVICES=""` to force CPU-only FL (slow but memory-safe).
