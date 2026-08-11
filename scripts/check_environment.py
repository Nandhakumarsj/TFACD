from __future__ import annotations

import platform
import sys

import torch

print("Python:", sys.version)
print("OS:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.T
    print("CUDA smoke test:", float(y.mean().cpu()))
else:
    print("WARNING: CPU mode. For Quadro P5000, install a CUDA 12.6 PyTorch wheel.")
