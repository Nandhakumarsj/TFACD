# TFACD pipeline runbook

Every stage below is a real, standalone script - this is the complete command
sequence from raw dataset to a running agentic + trust-boundary demo, with the
artifact each stage produces and what to inspect at each intermediate step.
README.md documents the "happy path" and the LLM decision engine specifically;
this runbook is the full reference, including the scripts README.md doesn't
walk through (Gate 4's benchmark, certification, evaluation, the dashboard).

## 0. Environment

```powershell
python scripts/check_environment.py
```
Prints Python/CUDA/PyTorch versions and runs a smoke matmul on the GPU if available. No artifact - a pass/fail check, run this first on any new machine.

## 1. Dataset inspection (no artifact required yet)

```powershell
python scripts/inspect_dataset.py --config configs/edge_iiot.yaml
```
Reads the raw CSV directly (no preprocessing yet). Prints schema, class distribution, leakage warnings (identifier-like columns, near-duplicate rows), and whether a defensible row ordering exists for temporal claims. **Read this output before changing `sequence_length` in the config** - see README's "Important scientific constraint".

## 2. Preprocessing

```powershell
python scripts/preprocess_dataset.py --config configs/edge_iiot.yaml
```
Produces `artifacts/data/prepared.npz` (train/val/test splits, already feature-encoded) and `artifacts/data/metadata.json` (`classes`, `input_columns`, `numeric_columns`, `categorical_columns`, `feature_dim`) plus `artifacts/data/preprocessor.joblib` (the fitted encoder/scaler - train-only fit, reused everywhere downstream: `training/`, `streaming/features.py`, the LLM benchmark's sample contexts do NOT need this, but the streaming pipeline does). **Inspect `metadata.json` directly** - it is the single source of truth `evaluate_checkpoint.py`, `streaming/pipeline.py`, and the benchmark scripts all read `classes`/`feature_dim` from.

## 3. Centralized baseline

```powershell
python scripts/train_centralized.py --config configs/edge_iiot.yaml
```
Produces `artifacts/models/centralized_best.pt` + a printed per-class classification report (accuracy/precision/recall/macro-F1/confusion matrix). This is the ceiling the federated experiments are compared against - **re-run `evaluate_checkpoint.py` (below) any time you want this number again**, rather than trusting a stale terminal scrollback.

## 4. Client partitioning

```powershell
python scripts/create_partitions.py --config configs/edge_iiot.yaml
```
Produces `artifacts/data/partitions/client_<N>.npy` (one index array per client, Dirichlet non-IID by default per `federated.partition_mode`/`dirichlet_alpha`). Required before any Flower run - both the live federation and Gate 4's standalone benchmark read these files.

**Steps 0-4 together are `scripts/run_first_milestone.ps1`** - a convenience wrapper, not a separate stage: `./scripts/run_first_milestone.ps1 -Config configs/edge_iiot.yaml` runs 0/1/2/3/4 in sequence and stops (deliberately does not proceed to Flower - "review metrics before running Flower" is printed at the end).

## 5. Federated training (simulation)

```powershell
flwr run . --stream
```
Wires `federated/server_app.py` + `federated/client_app.py` through Flower's simulation engine (single process, no real network sockets - see README's "Deployment-mode TLS" section for the separate real-network path). `use-ftil`/`proximal-mu`/`num-server-rounds` are read from `pyproject.toml`'s `[tool.flwr.app.config]` (override per-run: `flwr run . --run-config "num-server-rounds=5"`). Produces `artifacts/models/flower_ftil_final.pt` (or `flower_fedavg_final.pt` if `use-ftil=false`) plus its `.manifest.json`, and - only when FTIL is enabled - `artifacts/models/ftil_trust_log.jsonl`, one JSON line per round with per-client trust/OOD/clustering evidence.

**Read the FL trust evidence:**
```powershell
python scripts/summarize_ftil_trust_log.py
```
Human-readable per-round table (cluster method, explained variance, silhouette, accepted/rejected counts, max OOD score) plus a per-client EMA trust trajectory - see README's FL section for what each of `trust_scores`/`ood_scores`/`detection` actually measures and why they're different signals.

## 6. Checkpoint evaluation (any checkpoint, any time)

```powershell
python scripts/evaluate_checkpoint.py --checkpoint artifacts/models/flower_ftil_final.pt --label "FTIL round 3"
```
Loads a checkpoint against the held-out test split from `prepared.npz` and prints the same per-class report as Gate 2's centralized run - the **authoritative** accuracy number for any checkpoint, referenced by both `run_streaming_demo.py`'s subsample macro-F1 caveat and `test_streaming_pipeline.py`'s golden test. Run this after every training run, not just once.

## 7. Gate 4: FTIL attack/defense benchmark (standalone, not the live strategy)

```powershell
python scripts/run_integrity_benchmark.py --config configs/edge_iiot.yaml
```
Runs the attack x defense matrix (`label_flip`/`sign_flip`/`gaussian_noise`/`gradual_scaling`/`none` x `no_defense`/`pca_cluster_ema`/`coordinate_median`/`trimmed_mean`) in an isolated harness - **does not go through Flower's simulation engine or the live `IntegrityAwareStrategy`**, see the module docstring. Produces `artifacts/integrity/benchmark_report.json` (per-cell macro-F1/accuracy, detector TPR/FPR/TNR, aggregation overhead). This is what justified `trimmed_mean` as the live default (detector alone only caught ~33% of attacks; robust aggregation held ~0.93-0.94 macro-F1 regardless).

**Tune the detector against this benchmark:**
```powershell
python scripts/run_threshold_optimizer.py --config configs/edge_iiot.yaml
```
Narrow grid search over `reject_below_trust`/`ema_alpha` against Gate 4's labeled scenarios (`analytics/feedback_loop.py`). Read its own printed "ISOLATED-DETECTOR caveat" before trusting any TPR/FPR gain - it tunes the isolated detector under `weighted_average`, not the live `trimmed_mean`-based strategy.

## 8. Model certification (required before the streaming pipeline will load anything)

```powershell
python scripts/certify_model.py artifacts/models/flower_ftil_final.pt --sign
python scripts/verify_certified_model.py artifacts/models/flower_ftil_final.pt
```
`certify_model.py` promotes the manifest's `status` to `"certified"` and (with `--sign`) writes an Ed25519 signature to `artifacts/keys/`. **Required after every retraining run** - `server_app.py` rewrites the manifest (and therefore invalidates the old signature) on every training run, by design; see README's certification-state-machine notes. `verify_certified_model.py` is the same check `streaming/pipeline.py::StreamingIDS.from_config()` runs automatically before loading - run it manually any time you want to confirm a checkpoint is trustworthy without also running the full streaming demo.

## 9. Streaming pipeline + agentic + trust boundary (the full loop)

```powershell
python scripts/run_streaming_demo.py
```
Requires steps 2, 6/8 (a certified checkpoint) done first. Replays held-out rows through the certified IDS -> Threat Context Generator -> Agentic Decision Engine (template or LLM, per `agentic.decision_engine.engine`) -> Adaptive Semantic Trust Boundary -> capability execution -> audit log. Produces `artifacts/streaming/audit_log.jsonl` and `artifacts/streaming/history.jsonl`. See README's "LLM-backed Agentic Decision Engine" section for the extra steps (Ollama install, model pull, `run_llm_engine_benchmark.py` gate) needed before setting `engine: "llm"`.

`scripts/run_trust_boundary_demo.py` is a lighter-weight sibling - hand-scripted synthetic alerts plus a synthetic multi-agent population, no dependency on a real trained checkpoint at all. Use it to exercise the trust boundary in isolation.

## 10. Reporting

```powershell
python scripts/generate_security_dashboard.py --audit-log artifacts/streaming/audit_log.jsonl
```
Static HTML dashboard (acceptance rate, trust-level distribution, per-agent ranking) from any audit log - point `--audit-log` at either `artifacts/trust_boundary/audit_log.jsonl` (the synthetic demo) or `artifacts/streaming/audit_log.jsonl` (the real IDS-originated one); they are deliberately separate files, see README.

## 11. Phase II analytics reporting (all read a real audit log/history file - run step 9 or the trust boundary demo first)

```powershell
python scripts/run_trust_forecast.py --audit-log artifacts/trust_boundary/audit_log.jsonl
python scripts/run_drift_report.py --audit-log artifacts/trust_boundary/audit_log.jsonl --ftil-trust-log artifacts/models/ftil_trust_log.jsonl
python scripts/run_reputation_report.py --audit-log artifacts/trust_boundary/audit_log.jsonl
python scripts/run_explainability_report.py
```
`run_trust_forecast.py` - near-term trust/risk trajectory per agent (`analytics/trust_forecasting.py`). `run_drift_report.py` - Page-Hinkley drift detection over both independent data sources (agentic-side audit log and FL-side FTIL trust log - two different questions, reported separately, see README's FL metrics section). `run_reputation_report.py` - agents ranked best to worst (`analytics/reputation.py`). `run_explainability_report.py` - SHAP/LIME attribution for a representative well-formed vs. off-topic plan (cannot replay a real historical incident - the audit log doesn't retain enough of the original plan to reconstruct one exactly, see the script's own docstring).

```powershell
python scripts/refit_behavioral_trust.py --history artifacts/agentic/history.jsonl
```
Diagnostic only, not wired into the live pipeline: refits the Behavioral Trust Engine's IsolationForest from real observed history instead of its synthetic cold-start population, and prints a before/after comparison. Nothing is persisted - `boundary.py` is unaffected by running this.

## Full sequence, start to finish

```powershell
python scripts/check_environment.py
python scripts/inspect_dataset.py --config configs/edge_iiot.yaml
python scripts/preprocess_dataset.py --config configs/edge_iiot.yaml
python scripts/train_centralized.py --config configs/edge_iiot.yaml
python scripts/create_partitions.py --config configs/edge_iiot.yaml
flwr run . --stream
python scripts/summarize_ftil_trust_log.py
python scripts/evaluate_checkpoint.py --checkpoint artifacts/models/flower_ftil_final.pt
python scripts/certify_model.py artifacts/models/flower_ftil_final.pt --sign
python scripts/verify_certified_model.py artifacts/models/flower_ftil_final.pt
python scripts/run_streaming_demo.py
python scripts/generate_security_dashboard.py --audit-log artifacts/streaming/audit_log.jsonl
```

Optional, not required for the above: `run_integrity_benchmark.py` + `run_threshold_optimizer.py` (Gate 4 tuning, independent of the main sequence), `run_deployment_smoke_test.ps1` (real-network TLS deployment instead of simulation, see README), `run_llm_engine_benchmark.py` (required only if you intend to set `agentic.decision_engine.engine: "llm"`).
