# Phase-I implementation gates

## Gate 0 — Environment

- Python 3.11
- CUDA-enabled PyTorch reports `torch.cuda.is_available() == True`
- GPU name is Quadro P5000
- a tiny tensor operation succeeds on CUDA

## Gate 1 — Dataset audit

Deliverables:

- `artifacts/data/schema_report.json`
- label distribution
- missing-value report
- duplicate count
- candidate label/timestamp/group columns
- leakage warnings
- evidence on whether temporal ordering exists

Decision:

- If the selected DNN CSV is shuffled or lacks time/group order, set sequence length to 1.
- If time/order is defensible, create grouped sliding windows and compare sequence lengths.

## Gate 2 — Centralized baseline

- train-only preprocessing fit;
- macro-F1 and per-class recall;
- confusion matrix;
- no federated code yet;
- checkpoint saved.

Acceptance:

- training loss decreases;
- predictions cover more than one class;
- no NaN/Inf;
- minority-class metrics are reported.

## Gate 3 — Federation correctness

- 5 simulated clients;
- IID split first;
- FedAvg reproduces a sensible fraction of centralized performance;
- then Dirichlet non-IID split;
- compare FedAvg and FedProx.

## Gate 4 — Federated Trust & Integrity baseline

Attack simulations:

- label flipping;
- sign-flip/model-replacement update;
- Gaussian update noise;
- gradual scaling across rounds.

Defenses:

- finite/shape/range validation;
- PCA + clustering detector;
- historical EMA correction;
- coordinate median and trimmed mean comparisons.

Metrics:

- malicious-client TPR/TNR/FPR;
- clean global accuracy/F1;
- attacked global accuracy/F1;
- backdoor success rate only when a precise trigger experiment is implemented;
- aggregation overhead.

## Gate 5 — Certified model release

- global checkpoint;
- SHA-256 manifest;
- optional Ed25519 signature;
- verification before runtime loading.

## Gate 6 — Runtime plane

The certified model's prediction schema is the contract consumed by the Threat Context Generator.
