# TFACD second-review extension

Base repository reviewed: `Nandhakumarsj/TFACD`, GitHub `main` at commit `e027862df06aee0c7c1c472dc955fec3ad0f41d6` (2026-10-07). The extension is additive and keeps the existing runtime/ASTB path unchanged unless the explicit leakage-safe split configuration is enabled.

## Install into the repository

Copy this bundle so its `src/`, `scripts/`, `configs/`, `tests/`, and `docs/` directories merge into the TFACD repository root. Then run:

```powershell
python scripts/apply_second_review_extension.py
```

The installer is idempotent and refuses to edit a file when a source pattern does not match exactly.

Run the regression suite before generating new experiment artifacts:

```powershell
python -m pytest -q
```

## Important test failures from the current local checkout

The current GitHub main already contains the fixes for the two boundary failures and the real-Ollama audit-artifact failure. The remaining FTIL drift test needed to stop hard-coding Flower's old `0..4` node IDs. The installer applies that repair using the IDs actually present in the JSONL trust log.

## Order for the second-review experiments

### 1. Leakage-safe preprocessing

The installer changes `data.split_strategy` to `session_temporal` but deliberately leaves `sequence_length: 1`. This satisfies the current streaming/model input contract while preventing random row-level train/test mixing across the same flow/session.

Regenerate all data artifacts:

```powershell
python scripts/preprocess_dataset.py --config configs/edge_iiot.yaml
python scripts/validate_split_leakage.py --config configs/edge_iiot.yaml
```

Because the split changes, the old centralized/federated checkpoints are no longer scientifically comparable to the new test split. Re-run the training/evaluation sequence after this step.

### 2. ASTB-IIoT benchmark

Generate 1,000 scenarios:

```powershell
python scripts/generate_astb_benchmark.py --config configs/edge_iiot.yaml --count 1000 --seed 42
```

Run A–D and the requested component ablation:

```powershell
python scripts/run_astb_ablation.py --config configs/edge_iiot.yaml
```

The oracle is deterministic and independent of the LLM. Review `configs/astb_oracle.yaml` with the supervisor/domain reviewer and freeze it before publication.

### 3. FL security matrix

Create a separate ten-client partition set so 10%, 20%, and 30% mean exactly 1, 2, and 3 malicious clients:

```powershell
python scripts/create_partitions.py --config configs/edge_iiot.yaml --num-clients 10 --output-dir artifacts/data/partitions_10
```

Smoke test one attack first:

```powershell
python scripts/run_fl_security_matrix.py --config configs/edge_iiot.yaml --partitions-dir artifacts/data/partitions_10 --num-clients 10 --rounds 3 --attacks sign_flip
```

Then run the full matrix:

```powershell
python scripts/run_fl_security_matrix.py --config configs/edge_iiot.yaml --partitions-dir artifacts/data/partitions_10 --num-clients 10 --rounds 3
```

The three modes are:

- `fedavg`
- `fedprox`
- `ftil` (validation + PCA/cluster/EMA + trimmed mean)

The report includes accuracy, macro precision/recall/F1, binary attack detection rate, malicious-client detection rate, and global-model degradation relative to the clean condition.

### 4. External validation

IoT-23 is used as a cross-dataset binary attack-vs-benign validation because its Zeek flow schema and labels differ from Edge-IIoTset. This avoids inventing a false 15-class mapping.

After obtaining an IoT-23 `conn.log.labeled` file:

```powershell
python scripts/validate_external_iot23.py --csv "D:\datasets\IoT-23\CTU-IoT-Malware-Capture-34-1\bro\conn.log.labeled" --checkpoint artifacts/models/flower_ftil_final.pt
```

The report explicitly records feature coverage and warns that this is schema-adapted external validation.

### 5. Figures

After the result files exist:

```powershell
python scripts/plot_second_review_results.py
```

Outputs are individual PNG figures under `artifacts/second_review_figures/`, suitable for insertion into the Anna University report and for selecting compact figures for the 10–12 page LNCS paper.

## Metric definitions used by the extension

`FAR = unsafe actions incorrectly allowed / total unsafe actions`.

`UABR = unsafe actions correctly blocked / total unsafe actions`.

`SAPR = safe actions correctly allowed / total safe actions`.

The common safety interpretation is used: the FAR numerator counts the security failure, so it cannot be the other way around.

## Research honesty constraints

Do not claim that the policy oracle is a universal expert truth; it is a project-authored rule set that must be reviewed and frozen by a domain reviewer.

Do not claim the IoT-23 experiment is a matched 15-class evaluation. It is binary attack-vs-benign transfer through the current Edge-IIoTset preprocessing feature schema.

Do not call a malicious-client detection rate a model attack detection rate. The former is an FTIL/client-level signal; the latter is an IDS/test-set signal.
