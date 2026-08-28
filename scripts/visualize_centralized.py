import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from tfacd.common.config import load_config
from tfacd.data.preprocess import load_prepared
from tfacd.training.centralized import build_model


with open("artifacts/models/centralized_metrics.json",'r') as f:
    data = json.load(f)

# ==========================================
# 1. PLOT TRAINING & VALIDATION CURVES
# ==========================================
epochs = [entry["epoch"] for entry in data["history"]]

train_loss = [entry["train"]["loss"] for entry in data["history"]]
val_loss = [entry["validation"]["loss"] for entry in data["history"]]

train_acc = [entry["train"]["accuracy"] for entry in data["history"]]
val_acc = [entry["validation"]["accuracy"] for entry in data["history"]]

train_f1 = [entry["train"]["macro_f1"] for entry in data["history"]]
val_f1 = [entry["validation"]["macro_f1"] for entry in data["history"]]

plt.figure(figsize=(14, 5))

# Loss Plot
plt.subplot(1, 3, 1)
plt.plot(epochs, train_loss, label="Train Loss", marker="o")
plt.plot(epochs, val_loss, label="Validation Loss", marker="o")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 3, 2)
plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
plt.plot(epochs, val_acc, label="Validation Accuracy", marker="o")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Macro F1 Plot
plt.subplot(1, 3, 3)
plt.plot(epochs, train_f1, label="Train Macro F1", marker="o")
plt.plot(epochs, val_f1, label="Validation Macro F1", marker="o")
plt.title("Macro F1 over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ==========================================
# 2. PLOT CONFUSION MATRIX
# ==========================================
cm_data = data["test_confusion_matrix"]
labels = cm_data["labels"]
matrix = np.array(cm_data["matrix"])

plt.figure(figsize=(12, 10))
sns.heatmap(
    matrix,
    annot=False,  # Set to True if you want numbers inside cells (might look crowded)
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("Test Set Confusion Matrix")
plt.xlabel("Predicted Label") 
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


config = load_config("configs/edge_iiot.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prepared = load_prepared(config["data"]["output_dir"])

model = build_model(config, prepared.feature_dim, prepared.num_classes).to(device)
checkpoint = "artifacts/models/centralized_best.pt"
payload = torch.load(checkpoint, map_location=device, weights_only=True)
model.load_state_dict(payload["state_dict"])

model.eval()

# ==========================================
# 3. CHECK ARCHITECTURE
# ==========================================
print("=== Model Architecture (Standard) ===")
print(model)


# # # ==========================================
# # # 4. GRAPHICAL VISUALIZATION (TorchViz)

# try:
#     from torchviz import make_dot
    
#     dummy_input = torch.randn(config['training']['batch_size'], 1, prepared.feature_dim).to(device)
    
#     output = model(dummy_input)
    
#     dot = make_dot(output, params=dict(model.named_parameters()))
#     dot.format = "png"
#     dot.render("artifacts/models/model_architecture_graph")
#     print("\nGraphical visualization saved to `artifacts/models/model_architecture_graph.png`")
# except ImportError:
#     print("\nTip: To use torchviz, install it via `pip install torchviz` (and ensure system graphviz is installed).")