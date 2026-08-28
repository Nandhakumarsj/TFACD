# =============================================================================
# TFACD – Trustworthy Federated Agentic Cyber Defense for IIoT
# Multi-stage Dockerfile
#
# Stages
#   base     – common Python + CUDA base; all system deps
#   builder  – pip install (cached layer)
#   training – entry-point for the training plane (Flower FL + FTIL)
#   runtime  – entry-point for the runtime plane (streaming + agentic engine)
# =============================================================================

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04

# ---------------------------------------------------------------------------
# Stage 1 – base: Python 3.11 on CUDA 12.6
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        git curl ca-certificates \
        # nftables + iptables are the real firewall back-ends for ProductionExecutor
        nftables iptables iproute2 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2 – builder: install all Python deps into a venv
# ---------------------------------------------------------------------------
FROM base AS builder

COPY pyproject.toml ./
COPY src/tfacd/__init__.py src/tfacd/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

# Install the package with all optional groups
RUN pip install -e ".[dev,flower,agentic-llm,viz,explainability,mqtt]"

# ---------------------------------------------------------------------------
# Stage 3 – training: full training plane (FL simulation + FTIL)
# ---------------------------------------------------------------------------
FROM base AS training

ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

COPY . .

RUN mkdir -p artifacts/data artifacts/models artifacts/keys artifacts/agentic

CMD ["python", "scripts/train_centralized.py", "--config", "configs/edge_iiot.yaml"]

# ---------------------------------------------------------------------------
# Stage 4 – runtime: streaming + agentic decision engine
# ---------------------------------------------------------------------------
FROM base AS runtime

ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN mkdir -p artifacts/data artifacts/models artifacts/keys artifacts/agentic

EXPOSE 11434

CMD ["python", "scripts/run_streaming_demo.py", "--config", "configs/edge_iiot.yaml"]
