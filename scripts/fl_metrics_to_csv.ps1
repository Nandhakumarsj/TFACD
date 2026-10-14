@'
import json
from pathlib import Path
import pandas as pd

rows = []

central_path = Path("artifacts/models/centralized_metrics.json")
if central_path.exists():
    data = json.loads(central_path.read_text(encoding="utf-8"))
    m = data["test"]
    rows.append({
        "mode": "centralized",
        "attack": "clean",
        "malicious_clients_pct": 0,
        "effective_malicious_clients_pct": 0,
        "accuracy": m["accuracy"],
        "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"],
        "macro_f1": m["macro_f1"],
        "attack_detection_rate": None,
        "attack_detection_precision": None,
        "malicious_client_detection_rate": None,
        "macro_f1_degradation": None,
        "accuracy_degradation": None,
        "runtime_seconds": None,
    })

source = Path("artifacts/integrity/fl_security_matrix.json")
data = json.loads(source.read_text(encoding="utf-8"))

for mode, m in data["clean_baselines"].items():
    rows.append({
        "mode": mode,
        "attack": "clean",
        "malicious_clients_pct": 0,
        "effective_malicious_clients_pct": 0,
        "accuracy": m["accuracy"],
        "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"],
        "macro_f1": m["macro_f1"],
        "attack_detection_rate": m["attack_detection_rate"],
        "attack_detection_precision": m["attack_detection_precision"],
        "malicious_client_detection_rate": None,
        "macro_f1_degradation": 0.0,
        "accuracy_degradation": 0.0,
        "runtime_seconds": None,
    })

for r in data["results"]:
    m = r["metrics"]
    rows.append({
        "mode": r["mode"],
        "attack": r["attack"],
        "malicious_clients_pct": 100 * r["malicious_fraction_requested"],
        "effective_malicious_clients_pct": 100 * r["effective_malicious_fraction"],
        "accuracy": m["accuracy"],
        "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"],
        "macro_f1": m["macro_f1"],
        "attack_detection_rate": m["attack_detection_rate"],
        "attack_detection_precision": m["attack_detection_precision"],
        "malicious_client_detection_rate": r["malicious_client_detection_rate"],
        "macro_f1_degradation": r["global_model_degradation_macro_f1"],
        "accuracy_degradation": r["global_model_degradation_accuracy"],
        "runtime_seconds": r["runtime_seconds"],
    })

result = pd.DataFrame(rows)
output = Path("artifacts/integrity/fl_security_metrics.csv")
result.to_csv(output, index=False)
print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved: {output.resolve()}")
'@ | python -