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
pip install -e ".[dev,flower,agentic-llm]"
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

This produces `artifacts/models/flower_ftil_final.pt` and, when `use-ftil` is enabled (the default), `artifacts/models/ftil_trust_log.jsonl` - one JSON line per round recording each client's clustering-based accept/reject decision, EMA-smoothed personalized trust score, and continuous out-of-distribution (OOD) score, plus which clustering technique produced them. Read it with:

```powershell
python scripts/summarize_ftil_trust_log.py
```

which prints a per-round table (cluster method, PCA explained variance, silhouette, accepted/rejected/max-OOD) and a per-client trust trajectory. See "Federated Trust & Integrity Layer metrics" below for what each signal actually measures.

**A full command-by-command reference for every stage and intermediate artifact, including the scripts not covered above (Gate 4's attack benchmark, certification, checkpoint evaluation, the dashboard) is in [docs/pipeline_runbook.md](docs/pipeline_runbook.md).**

## Federated Trust & Integrity Layer metrics

`integrity/detector.py::PCAClusterEMAFilter.detect()` returns three distinct per-client signals - they answer different questions, and reporting one in place of another overstates what was measured:

- **`trust_scores`** (0..1) - EMA-smoothed personalized client trust, the only one of the three that carries across rounds. Gate 4 measured that a single bad round does not sink a client here by design (needs several consistent rounds to cross `reject_below_trust`), which is why robust aggregation (`trimmed_mean`) is kept as defense-in-depth rather than relying on this alone.
- **`ood_scores`** (>=0, unbounded) - this round's distance from the cohort's robust center in PCA space, as a multiple of the cohort's *median* distance (1.0 = typical, 2.0 = twice as far out as the median client). Memoryless and continuous, so it still ranks clients even when clustering collapses everyone into one bucket - which is exactly when `round_scores`/`trust_scores` alone convey nothing. Reported as `0.0` for a degenerate cohort (all-identical updates, e.g. late in convergence) rather than dividing by a ~zero distance.
- **`metrics`** (`DetectionMetrics`) - which clustering technique produced the round's decision (`cluster_method`: `"agglomerative"` or `"dbscan"`, or `"none:too-few-clients"` when fewer than 3 clients participate and clustering is skipped entirely), PCA explained variance ratio, silhouette score (`None` when not defined - DBSCAN can legitimately produce a single label), and whether the "too few clients flagged benign" distance-fallback path fired.

All three are persisted every round in `ftil_trust_log.jsonl` and surfaced in the live Flower run's `MetricRecord` (`ftil_explained_variance_ratio`, `ftil_silhouette`, `ftil_max_ood_score` - `cluster_method` itself is a string and Flower's `MetricRecord` only accepts numeric/list-numeric values, so it stays in the JSONL log rather than the per-round Flower metrics).

## Deployment-mode TLS (optional, not the default path)

Every command above uses Flower's local-simulation engine (single process, no real network sockets) - there is nothing to encrypt there. `scripts/run_deployment_smoke_test.ps1` is a separate, real verification path: it starts an actual `flower-superlink` + two `flower-supernode` processes as distinct OS processes talking over real TLS-encrypted sockets on `127.0.0.1`, and submits a live training round against them.

This is **not** "mTLS" - `flower-supernode --root-certificates`'s own docstring says "This is NOT a client certificate for mTLS." What's actually running is server-authenticated TLS (a local CA + SuperLink server certificate, `src/tfacd/security/certificates.py`) plus a separate SuperNode public-key node-authentication mechanism (EC/OpenSSH keypairs - **not** this repo's Ed25519 model-signing format, Flower requires a different key shape for node auth). Simulation stays the default day-to-day path; this is an alternate, tested one.

```powershell
python scripts/generate_deployment_certs.py    # CA + server cert, SuperNode auth keypairs -> artifacts/certs/
./scripts/run_deployment_smoke_test.ps1         # starts the processes, registers nodes, submits a real run, tears everything down
```

Manual sequence, if reproducing by hand: generate certs -> start `flower-superlink --ssl-certfile ... --ssl-keyfile ... --ssl-ca-certfile ... --enable-supernode-auth` -> `flwr supernode register <pubkey> local-tls` per node (returns a `node_id`, requires the SuperLink already running) -> start each `flower-supernode --auth-supernode-private-key ... --root-certificates ... --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:909<N>` (each SuperNode needs a distinct ClientAppIo port when running more than one on the same machine) -> `flwr run . local-tls --stream` (the SuperLink connection is a *positional* argument, not `--federation` - that flag is for Flower's hosted/cloud federation IDs in `@account/name` form, unrelated to selecting a local connection).

## Live attack-scenario streaming (optional, not the default path)

`scripts/run_streaming_demo.py` replays one flattened CSV (`DNN-EdgeIIoT-dataset.csv`) in a single batch. `./datasets/Edge_IIoTset` also has the **raw, per-attack-type and per-sensor-type CSV+PCAP capture pairs** that flattened file was itself assembled from (`Attack traffic/<Type>_attack.csv`+`.pcap`, `Normal traffic/<Sensor>/<Sensor>.csv`+`.pcap`) - the same 63-column schema, so they're structurally compatible with the existing feature pipeline already (`StreamingFeatureExtractor` selects columns by name, not file identity). `streaming/live_source.py` + `streaming/scenario.py` use these to build a genuinely **time-paced, multi-source** input stream instead of a flat replay - a normal-traffic baseline with a real attack file injected mid-stream, this project's answer to "there is no live MQTT/Modbus sensor to stream from."

**Why PCAP timestamps drive pacing, not the CSV's own `frame.time` column:** measured directly against real files in this repo - `Normal traffic/Modbus/Modbus.csv`'s `frame.time` column is **empty on every row**; only the paired `Modbus.pcap` carries real capture timing. Attack-traffic CSVs do have a populated `frame.time`, but `streaming/pcap_timing.py` (stdlib `struct` only, no scapy/pyshark/dpkt - reads only the 24-byte global header and each packet's 16-byte record header, never packet content) is used uniformly for both, since it's the only source that works on every file. CSV rows and PCAP packets are paired positionally, in capture order - justified by measured >99% row/packet-count alignment, documented as an approximation, not an exact guarantee.

```powershell
python scripts/run_attack_scenario.py --scenario portscan_source_block
python scripts/run_attack_scenario.py --scenario ddos_segment_isolation
python scripts/run_attack_scenario.py --scenario portscan_source_block --speed 1.0   # true real-time pacing, not compressed
```

`configs/attack_scenarios.yaml` defines the scenarios declaratively (baseline source(s), attack file, `speed_multiplier`, an `expected_capability` that's reported, never forced). Two attack-response scenarios reuse **existing, already-whitelisted** capabilities - no new capability strings were added to `configs/trust_policy.yaml`:

| scenario | injected attack | "closest real capability" |
|---|---|---|
| `portscan_source_block` ("IP isolation") | `Port_Scanning` | `block_source` |
| `ddos_segment_isolation` ("network disconnect") | `DDoS TCP SYN Flood` | `isolate_segment` |

This is simulated **input pacing**, not simulated **response** - whether a capability actually executes for real is entirely `trust_boundary.executor.mode`'s decision (see "Real response execution" below), never this script's.

**A third scenario, `modbus_normal_traffic_generalization_check`, exists to surface a finding, not to demo a clean success:** measured directly (n=100 rows each) - every Normal-traffic sensor type in this dataset (`Temperature_and_Humidity`, `Soil_Moisture`, `Distance`, `Water_Level`, `phValue`, `Sound_Sensor`, `Flame_Sensor`, `Heart_Rate`, `IR_Receiver`) classifies **100% correctly as `Normal`** through the certified model - except real Modbus traffic, which classifies as `Normal` only **~21% of the time** (59% misclassified `DDoS_UDP`, 20% `Port_Scanning`). Given this project's explicit IIoT/SCADA framing, a model that mostly fails to recognize real Modbus traffic as benign is a genuine, worth-stating generalization gap - the two attack-response scenarios above deliberately use `Temperature_and_Humidity` as their baseline instead (which does classify correctly) so their normal-vs-attack contrast is legible, rather than quietly avoiding Modbus everywhere. Run `python scripts/run_attack_scenario.py --scenario modbus_normal_traffic_generalization_check --max-incidents 100` to reproduce this yourself.

PCAP **packet content** is never parsed - only capture timestamps, for pacing. Rebuilding this dataset's flow-feature-extraction methodology from raw packets (to derive NEW features rather than reuse the paired CSV's) would be a substantial, error-prone undertaking with no existing dependency or reference implementation in this repo; the CSVs remain the sole source of feature values.

## Real response execution (optional, not the default path - "simulate" always is)

Every response action was, until now, always simulated: `SimulatedExecutor` (`trust_boundary/capability_enforcement.py`) only ever logs "would execute" and returns success - its own docstring called this "a deployment-specific extension point, not something this repo pretends to have." `trust_boundary/production_executor.py::ProductionExecutor` fills that seam: real OS-level actions, gated behind an explicit config switch that defaults to the old behavior.

```yaml
trust_boundary:
  executor:
    mode: "simulate"    # default, unchanged behavior. "production" takes real action - see below.
    protected_targets: ["127.0.0.1/32", "169.254.0.0/16", "::1/128"]  # hard rail, add your real
                                                                        # management/gateway addresses
```

| capability | Linux (primary real backend) | Windows (dev-workstation backend) |
|---|---|---|
| `block_source` | `nftables` (self-contained `tfacd` table/chain), `iptables` fallback if `nft` isn't installed | `netsh advfirewall firewall add rule ... action=block` |
| `isolate_segment` | same as above, against the action's subnet/CIDR | same as above, against the subnet/CIDR |
| `rate_limit` | `nftables`/`iptables` rate-limit rule, best-effort | **no backend** - no reliable built-in Windows CLI equivalent; falls back to simulated logging for this one capability, with a loud log line, never a silent no-op |
| `rotate_session` | real, but application-level: appends a timestamped rotation-request record (`artifacts/trust_boundary/session_rotation_requests.jsonl`) - this project has no live session-issuance service to rotate synchronously against (every `SessionContext` here is already freshly minted per call) |
| low-risk (`observe`, `log_event`, `increase_logging`, `create_ticket`, `notify_soc`, `start_capture`) | real, persisted log entry (`artifacts/trust_boundary/production_action_log.jsonl`), no OS action |

**Honest limitation, not silently worked around:** `isolate_segment`'s target is `alert.target_asset`, an asset **name** (e.g. `"plc-01"`) in this project's data model - there is no asset-name-to-IP/CIDR inventory anywhere in this repo. `ProductionExecutor` refuses cleanly (returns `False`, logs why) rather than fabricate a mapping, whenever a network-acting capability's target isn't already a parseable IP/CIDR.

**Correctness fix this workstream required, found by re-reading the actual code rather than assumed:** `agentic/decision_engine.py` previously handed **every** capability - including `block_source`/`rate_limit`, which act on the attacker - `alert.target_asset` (the protected device) as `target`. Harmless while only `SimulatedExecutor` ever read it; a real bug once a real executor needs to know *what* to block. Fixed: `block_source`/`rate_limit` now get `alert.source_id`; `isolate_segment` and every other capability keep `target_asset`, unchanged. The LLM-backed engine's system prompt got the same instruction added.

`os.detection` (`platform.system()`) picks the backend automatically - no config needed beyond `mode: "production"`. Verify on your own hardware before trusting either backend, the same discipline this README already applies to the LLM benchmark gate.

**Windows backend: live-fire verified end to end on this project's own dev workstation**, from an elevated (Administrator) terminal, against `203.0.113.1` (RFC 5737 - reserved for documentation/testing, guaranteed non-routable, so this proves the mechanism with zero possibility of affecting real traffic):

```powershell
.\.venv\Scripts\python.exe -c "from tfacd.runtime.contracts import CyberAction; from tfacd.trust_boundary.production_executor import ProductionExecutor; e = ProductionExecutor(); a = CyberAction(capability='block_source', target='203.0.113.1'); print('execute() ->', e.execute(a))"
netsh advfirewall firewall show rule name=tfacd_block_source_203.0.113.1   # confirm the rule really exists
netsh advfirewall firewall delete rule name=tfacd_block_source_203.0.113.1
```

Measured result: `execute()` returned `True`, `returncode: 0` in the recorded action log, `netsh ... show rule` confirmed `RemoteIP: 203.0.113.1/32`, `Action: Block`, `Direction: In` exactly as constructed, and delete succeeded. A non-elevated terminal correctly fails closed instead (`execute()` returns `False`, logged as `returncode: 1` - "requires elevation" from Windows itself, not a silent success) - real command construction and honest failure reporting were verified first, from a normal terminal, before re-running elevated.

**Linux backend: unit/command-construction-tested only** (see `tests/test_production_executor.py`) - no Linux environment exists on this dev workstation to live-fire it here. Verify on your own Linux target before trusting it:

```bash
sudo nft list tables                      # confirm nftables is available (or plan on the iptables fallback)
# then, with trust_boundary.executor.mode: "production" set locally, run a scenario and confirm:
sudo nft list table inet tfacd            # the rule really exists
sudo nft delete table inet tfacd          # clean up afterward
```

## LLM-backed Agentic Decision Engine (optional, not the default path)

Two decision engines exist, selected by `agentic.decision_engine.engine` in `configs/edge_iiot.yaml`:

- **`"template"`** (`agentic/decision_engine.py`) - deterministic. Proposes every playbook `threat_context.yaml` allows for that severity and fills one of five rationale templates. Fully reproducible; no external service.
- **`"llm"`** (`agentic/llm_engine.py` + `agentic/graph.py`) - a LangGraph reason -> validate -> retry loop over a local Ollama model. It reads the threat context and **selects a reasoned subset** of the allowed playbooks with a rationale grounded in the specific evidence.

Both satisfy the same `DecisionEngine` protocol (`agentic/base.py`) and produce a `CyberActionPlan` that goes through the **same unmodified Adaptive Semantic Trust Boundary**. The trust boundary has no idea which engine produced a plan - that separation is the point: the LLM proposes, the boundary decides.

```powershell
# 1. Install Ollama and pull a model (verify the size ollama itself reports; do not trust a number here)
winget install --id Ollama.Ollama
ollama pull qwen3:8b
# or, the best-measured option on this project's hardware (see the comparison table below):
ollama pull gemma3:4b

# 2. Install the optional dependency group
pip install -e ".[agentic-llm]"

# 3. REQUIRED: pass the benchmark gate before the engine can be used at all
python scripts/run_llm_engine_benchmark.py --model qwen3:8b
# gemma3:4b does not support Ollama tool-calling - it MUST use json_mode, or the gate fails outright:
python scripts/run_llm_engine_benchmark.py --model gemma3:4b --structured-output-method json_mode

# 4. Only then set agentic.decision_engine.engine: "llm" in configs/edge_iiot.yaml
#    (and agentic.llm.structured_output_method: "json_mode" if using gemma3:4b)
python scripts/run_streaming_demo.py
```

### Switchable main/fallback model (e.g. moving this codebase to a Quadro P5000)

`agentic.llm.model` (primary) and `agentic.llm.fallback_model` (optional, `null` by default) are two independent config keys, both requiring real Ollama tool-calling (`structured_output_method: "function_calling"` - this is not a `json_mode` escape hatch). If the primary model's graph exhausts its retries or errors, `LLMDecisionEngine` retries the *same* reason/validate/retry logic against the fallback model before dropping to the deterministic template engine - a three-tier chain (`engine` field records which: `"llm"`, `"llm:fallback_model"`, or `"fallback:<reason>"`). This is a **model-level** fallback, distinct from the engine-level `llm -> template` fallback that already existed.

```powershell
# Benchmark BOTH models in one run - writes {"primary": {...}, "fallback": {...}} instead of the
# flat single-model report shape, so the second model no longer silently clobbers the first's report:
python scripts/run_llm_engine_benchmark.py --model gemma4:12b --fallback-model qwen3:8b

# Only after that report says primary.go_no_go.go: true (and fallback.go_no_go.go: true, if you want
# the fallback tier wired in) do you edit the local config:
#   agentic.llm.model: "gemma4:12b"
#   agentic.llm.fallback_model: "qwen3:8b"
python scripts/run_streaming_demo.py
```

**Gemma 4** (Google, released April 2026 - confirmed against Ollama's own library listing and Google's developer blog, not assumed) ships in `e2b`/`e4b`/`12b`/`26b`/`31b` sizes with real tool-calling support, unlike Gemma 3. `--model`/`--fallback-model` and `structured_output_method` are plain config values everywhere in this codebase - neither is ever inferred from a model name - so `run_llm_engine_benchmark.py` needs no code change to benchmark `gemma4:12b`/`gemma4:26b`, only `ollama pull gemma4:12b` first. **What this README does not do: claim gemma4 passes the gate on your hardware.** Only your own benchmark report decides that - see the P4000/P5000 table below, which follows the identical discipline for the numbers that already exist here.

The checked-in `configs/edge_iiot.yaml` default stays `agentic.llm.model: "qwen3:8b"`, `fallback_model: null` - the only model with an actual passing report on this project's own hardware. Flipping either key is a two-line edit once your own run says `go: true`; nothing here ships an unverified default.

**The benchmark gate is a real precondition, not advice.** `agentic/factory.py::build_decision_engine` reads `artifacts/agentic/llm_benchmark_report.json` and refuses to construct the LLM engine unless it says `go: true` - the same "verify before load" discipline `verify_release()` applies to the certified checkpoint. It measures VRAM delta (via `nvidia-smi`, since Ollama runs as its own process and `torch.cuda.memory_allocated()` cannot see it), tokens/sec, p50/p95 latency, and JSON-schema validity both raw and after retry, across one representative alert per real attack class.

Measured on this project's actual workstation (Quadro P4000, 8GB, Pascal - **no tensor cores**), `qwen3:8b` at `num_ctx: 4096`:

| metric | measured | threshold |
|---|---|---|
| VRAM delta | 5476 MB | <= 85% of free VRAM at load time |
| tokens/sec | 22.1 | >= 5 |
| latency p50 / p95 | 29.8s / 39.5s | p95 <= 60s |
| schema validity (raw / after retry) | 1.00 / 1.00 | >= 0.80 / >= 0.95 |

### Running on a different GPU (e.g. Quadro P5000, 16GB)

`scripts/run_llm_engine_benchmark.py` auto-detects the installed card's total VRAM via `nvidia-smi` (`agentic/benchmark.py::nvidia_smi_total_vram_mb`) - it is not hardcoded to this project's 8GB P4000, and needs no config change to run on a different card. Pass `--total-vram-mb` only to override detection.

**This project's own workstation has a P4000, not a P5000 - the numbers below are an extrapolation from the measured P4000 behavior, not a measurement.** Re-run `scripts/run_llm_engine_benchmark.py` on the actual card and trust that report over this table.

The P4000 and P5000 are the same Pascal generation (no tensor cores either way) - the P5000 has ~43% more CUDA cores (2560 vs 1792), ~19% more memory bandwidth (288 vs 243 GB/s), and double the VRAM (16GB vs 8GB). Expect noticeably lower latency and much more VRAM headroom, not a different architecture-level tradeoff:

| setting | this project's P4000 (measured) | a P5000, extrapolated (unmeasured) |
|---|---|---|
| `agentic.llm.num_ctx` | `4096` (VRAM-constrained) | `8192`+ is affordable - 16GB has ~3x the headroom this model needs |
| `agentic.llm.model` | `qwen3:8b` | `qwen3:8b` still fits easily; `qwen3:14b` (~9-10GB Q4) becomes viable and is worth benchmarking for quality |
| VRAM gate | passes at 5476MB/8192MB (67%) | expect well under the 85% safety margin even with a larger model |
| latency | p50 29.8s / p95 39.5s | likely lower given more cores/bandwidth, but not measured - do not assume a specific number |

**gemma4 as a P5000 opt-in:** `gemma4:12b` (~8GB dense) is a reasonable P5000 main-model candidate to benchmark; `gemma4:26b` is a 26B MoE model (~4B active) whose actual quantized VRAM footprint against the 85%-of-free-VRAM gate is genuinely uncertain from a spec sheet alone - it may or may not fit even a 16GB card once KV cache/context overhead is included. Run `python scripts/run_llm_engine_benchmark.py --model gemma4:26b --fallback-model qwen3:8b --total-vram-mb 16384` (or omit `--total-vram-mb`, auto-detection works on a P5000 too) and let the report - not this table - decide. `qwen3:8b` as `fallback_model` is a safe choice on either card: it already has a passing report on the smaller P4000, so it's essentially guaranteed to fit a P5000's extra headroom too.

### Research honesty

- **The latency ceiling was revised after measuring, and that is recorded rather than hidden.** It started at 30s (an unmeasured guess) and was raised to 60s once the real p95 came in at ~40s. It is still a real gate - a model 50% slower than what was measured fails it. `~40s/decision` bounds a `max_incidents: 10` run at 5-7 minutes of LLM time, against an IDS stage that scores 20,000 records in 14s. The LLM is deliberately nowhere near that hot path.
- **`num_ctx: 4096`, not 8192** - a real decision uses ~534 prompt + ~703 output tokens, and halving the context window cut VRAM 6041MB -> 5438MB. That was the difference between failing and passing the VRAM check: an engineering fix, not a relaxed threshold.
- **Smaller is not automatically better, and it depends entirely on the model family.** All three measured on this box, official `run_llm_engine_benchmark.py` reports (git-ignored, re-run rather than trusting this table):

  | model | structured_output_method | VRAM delta | tokens/sec | p50 / p95 latency | raw / retry validity | go |
  |---|---|---|---|---|---|---|
  | `qwen3:8b` | `function_calling` | 5476 MB | 22.1 | 29.8s / 39.5s | 1.00 / 1.00 | **yes** |
  | `qwen3:4b` | `function_calling` | 2984 MB | 33.8 | 91.3s / 223.5s | 0.73 / 0.93 | no |
  | `gemma3:4b` | `json_mode` | **3733 MB** | **34.2** | **5.9s / 14.7s** | 0.87 / 1.00 | **yes** |

  `qwen3:4b` fails on latency and validity despite lower VRAM and comparable raw tokens/sec to `gemma3:4b` - it emitted far more tokens per decision (thinking-mode-style verbosity) and burned retries. A conciseness-directive system prompt and Qwen's `/no_think` marker were both tried against it directly (not through the benchmark gate) and made things *worse*, not better - both variants failed to parse where the unmodified baseline had succeeded on the same sample. This looks like a property of the model at this size, not something prompting fixes.

  `gemma3:4b` cannot use the default `function_calling` method at all - Ollama returns `registry.ollama.ai/library/gemma3:4b does not support tools (status code: 400)`, a hard capability gap, not a reliability problem. Under `agentic.llm.structured_output_method: "json_mode"` (schema spelled out in the prompt text instead of via tool-calling metadata - `agentic/graph.py::build_system_prompt`, `ChatOllama(format="json")`) it passes cleanly with the best numbers of any model tested on this hardware: smallest VRAM footprint, highest throughput, and by far the lowest latency. If reproducing this benchmark, try `gemma3:4b` with `json_mode` before assuming a larger model is necessary.
- **Switching engines changes the trust scores, and that is the finding, not a regression.** On the same 10 replayed incidents, the template engine reached `verified` trust and executed all 4 playbooks; the LLM engine reached `high`/`medium` and executed fewer. The cause is measurable: the template engine scores `Rs = 0.000` because `semantic_risk.py` compares a plan's rationale against the very template that produced it, and `Rc = 1.000` because the plan copies `alert.confidence` verbatim into `plan.confidence`, which is exactly what `context_consistency.py` checks alignment against. **Both signals were tautological while the deterministic engine was the only plan producer.** An LLM-authored plan (`Rs = 0.17-0.19`, `Rc = 0.85`) is the first real measurement either has produced here: the rationale is independently written, and the model sets its own calibrated confidence (0.65 against a detector that was only 0.343 confident). The resulting discount is defensible security behavior - an agent whose claims outrun its evidence *should* score lower - but it is a trust *floor* comparison, not an apples-to-apples one, and any evaluation that reports both engines' trust values side by side must say so.
- **Non-determinism.** Even at `temperature: 0.0`, GPU kernel scheduling and batching make local inference not bit-for-bit reproducible run to run. This is why the deterministic engine is kept as a selectable peer, not deleted - it is the reproducible baseline. Published Qwen3 benchmarks are for full-precision weights; these are Q4_K_M quantized.
- **The `engine` field is provenance, not a security control.** Every `CyberActionPlan` and `TrustDecision` carries `engine` (`"template"`, `"llm"`, or `"fallback:<reason>"`), so the audit log distinguishes an LLM-authored decision from one the fallback produced. It is unsigned metadata - a workflow/audit aid, not tamper protection.
- **The LLM can only ever narrow, never widen.** It selects from `context.allowed_playbooks` and nothing else; `trust_boundary/deterministic_controls.py` re-checks that independently, and `capability_enforcement.py` re-checks the whitelist again immediately before execution. If the model proposes an unlisted capability, the graph retries with the error, then falls back to the deterministic engine - so a plan that leaves this engine is never less constrained than one the template engine would have produced. Note the corollary: a playbook the model declines to propose can never execute later, even at high trust.
- **Qwen3's context is 32K natively**, not 128K - 128K requires YaRN extension, which Qwen's own documentation advises against enabling unless genuinely needed.
- RAG over a security corpus and the "with-vs-without ASTB" comparison experiment are **not implemented** and are deliberately out of scope for this pass.

## Analyst feedback loop: labeling trust decisions (optional, not the default path)

No ground truth for "this trust decision was actually wrong" existed anywhere on the agentic side (`analytics/feedback_loop.py`'s own docstring explains why - it's a *different* feedback loop, grid-searching FTIL's FL-side detector against a labeled attack benchmark that already exists; it deliberately never touches `trust_boundary/`). `analytics/trust_labels.py` + `analytics/threshold_validation.py` are the missing mechanism: any analyst can attach a label to a specific past `AuditEntry` (referenced by its `sequence`, the only stable id an entry already has).

```powershell
# Look up and label a specific past decision (prints the full AuditEntry first, so you see
# exactly what you're labeling before committing):
python scripts/label_trust_decision.py --audit-log artifacts/streaming/audit_log.jsonl --sequence 42 `
    --label false_positive --analyst-id "your-name" --rationale "block_source fired on benign traffic"

# Once enough labels have accumulated (>= 20 distinct labeled decisions), report agreement
# between analyst labels and trust_level_thresholds' actual behavior - reports only, never
# auto-tunes a live safety threshold:
python scripts/run_trust_threshold_validation.py
```

No login/user-account system exists anywhere in this codebase - `--analyst-id` is free text. "Anyone can label" is the point of this mechanism, not an oversight, matching this project's current single-workstation trust model. The label store is hash-chained (reuses `chain_hash()` from `integrity/certification.py`, the same primitive `AuditLogger` uses) so a tampered label is as detectable as a tampered audit entry, without a second crypto scheme.

**This becomes more important, not just a research nicety, once `trust_boundary.executor.mode: "production"` is set** (see "Real response execution" below): a labeled decision whose `executed_actions` came from `ProductionExecutor` rather than `SimulatedExecutor` is flagged as the highest-priority row in `run_trust_threshold_validation.py`'s report, since it already had a real-world effect, not just a logged recommendation.

**What this does NOT claim:** the day this mechanism ships, `trust_level_thresholds` are still unvalidated - they stay unvalidated until real labels actually accumulate. Building the mechanism is not the same claim as having validated the thresholds; see "Known gaps" below.

## Known gaps (2026-08-13 architecture audit, closed out 2026-08-14)

A structured audit (six dimensions, every "bug"/"gap" claim independently adversarially re-verified against the actual code before being recorded - all survived re-verification) found the items below. All but one are now fixed; the remainder is a genuine research limitation, not a patchable bug.

**Fixed:**
- `training.class_weighting: true` was applied in the centralized baseline but silently dropped in the federated path (`federated/client_app.py` built an unweighted `CrossEntropyLoss()` regardless of config). Fixed: each client now weights by its own local partition's class distribution (not the global one) via `_weighted_criterion()`.
- `IDSAlert.source_id`/`target_asset` flowed unsanitized into the LLM decision engine's prompt and into `semantic_risk.py`'s scoring - an unmitigated prompt-injection surface. Fixed at the source: `streaming/pipeline.py::_alert_for` now canonicalizes both fields (Unicode NFKC + zero-width stripping) before constructing the alert.
- **Nonce replay** - `preprocessing.py` now rejects a reused `(agent_id, nonce)` pair within the session freshness window, backed by a new `EntityHistory` event kind. Fixing this surfaced a real second bug: `agentic/synthetic.py` and both demo scripts previously reused a single nonce across multiple real `evaluate()` calls (harmless before the check existed, a hard rejection after) - all three now mint a genuinely unique nonce (`uuid4()`) per call, not one derived from a fixed name/index that would collide with the *same script's own prior run* against its persisted history file.
- **Leetspeak detection** - `preprocessing.py` now flags a substitution-hidden dangerous keyword (`"1gn0r3"` → `"ignore"`) via a narrow, keyword-anchored de-leet check, deliberately not a general "any digit near a letter" heuristic (verified empirically against device names like `"gateway-01"` and `"vlan10"` - no false positives).
- **`dirichlet_partition`'s per-class blind spot** - a new opt-in `min_class_samples` guard (wired into `create_partitions.py` at `min_class_samples_per_client: 1`) requires every client to see at least that many examples of every class. Verified against the real dataset: the exact alpha/seed combination that previously dropped Fingerprinting to 0 samples on two clients now guarantees ≥1 everywhere.
- **`federated/loaders.py` redundant I/O** - `client_loaders()` now caches the decompressed `prepared.npz` (`functools.lru_cache`, keyed on `output_dir`) instead of reloading it on every Flower message.
- **`capability_enforcement.enforce()`'s incomplete re-check** - now takes `context: ThreatContext` and re-verifies `context.allowed_playbooks` immediately before execution, not just the static whitelist.
- **Semantic Risk TF-IDF fallback's spurious penalty** - vocabulary is now enriched from `threat_context.yaml`'s 15 real classes (both identifier-style and natural-language-phrased renderings, since a real LLM never writes the literal capability string `block_source`). Corpus tuning alone wasn't enough on its own to fully close the gap (word-choice variance is a structural bag-of-words limitation, not a fittable vocabulary problem) - the actual fix is a targeted floor: if the rationale names the real attack type, the TF-IDF-derived risk is capped at a moderate ceiling rather than left to fall wherever word overlap happens to land. Verified this floor does *not* rescue a rationale naming the *wrong* attack type.
- **Behavioral Trust Engine now adapts** - `BehavioralTrustEngine.refit_from_history()` reconstructs real observed feature vectors from `EntityHistory`'s stored `trust_decision` events (exact reconstruction of `high_risk_fraction`/`action_count` from stored capabilities, exact retrospective replay of the 1-hour `violation_rate`/`recent_event_count` window) and refits the IsolationForest once ≥20 real samples exist. Exposed via `scripts/refit_behavioral_trust.py` - deliberately not auto-wired into the live pipeline (no persistence path exists for the refit forest; this is an operator-run diagnostic, same posture as `run_threshold_optimizer.py`).
- **5 of 7 Phase II analytics modules had no script entry point** - now all 7 do: `scripts/run_trust_forecast.py`, `scripts/run_drift_report.py`, `scripts/run_reputation_report.py`, `scripts/run_explainability_report.py` join the pre-existing threshold-optimizer and KPI-dashboard scripts. All four verified against real accumulated history in this repo, producing genuinely sensible output (e.g. the reputation ranking correctly places `agent-well_behaved` first and `agent-risky` last).

**Not fixed - a research limitation, not a bug:** no script validates `trust_level_thresholds` (0.40/0.65/0.85) against real labeled outcomes. This can't be patched the way the items above could: `feedback_loop.py`'s own docstring explains why - no ground truth exists anywhere for "this trust decision was actually wrong" (unlike the FL-side detector, which Gate 4's labeled attack benchmark *can* validate against). Closing this would require building an analyst-feedback labeling mechanism first, a separate, larger scope than a fix.

## Known gaps (2026-08-20 follow-up pass)

**Fixed:**
- **Analyst feedback loop / trust-decision labeling mechanism now exists** (`analytics/trust_labels.py`, `analytics/threshold_validation.py`, `scripts/label_trust_decision.py`, `scripts/run_trust_threshold_validation.py` - see "Analyst feedback loop" above). This closes the "labeling mechanism must exist first" precondition the 2026-08-14 audit called out - `trust_level_thresholds` themselves remain **unvalidated until real labels actually accumulate**, which is a separate, ongoing claim from "the mechanism exists," not the same one.
- **`plan.rationale` obfuscation-check gap** - `trust_boundary/preprocessing.py` ran the base64/hex/URL-encoding/leetspeak obfuscation check over every string `action.parameter`, but never over `plan.rationale` itself, despite canonicalizing it - the field most exposed to an LLM-authored obfuscated instruction-override payload. Fixed: rationale now runs through the same existing check.
- **Private key storage hygiene** - `security/certificates.py` and `integrity/signing.py` wrote every private key (CA, server, SuperNode auth, Ed25519 signing) with no OS permission hardening. Fixed: `os.chmod(0o600)`-equivalent applied after each write - a no-op beyond the read-only bit on this project's Windows dev box, real protection on the Linux IIoT edge devices that are this project's actual deployment target.
- **`certification.py::verify_release`'s `signature_ok is not False` boolean check** treated an unperformed signature check (`None`) as passing regardless of `require_signature`. Not exploitable through any real caller today (every default is `require_signature=True`), but the condition now spells out the actual safe invariant (`signature_ok is True or (signature_ok is None and not require_signature)`) instead of relying on an unstated one holding forever.
- **Dead config key removed** - `agentic.decision_engine.fallback_engine: "template"` was never read anywhere in the codebase (confirmed via full-repo search). Removed rather than left as a config value that silently did nothing; the engine-level `llm -> template` fallback it looked like it controlled is not actually configurable (it's the one architecturally-fixed fallback tier) - see "Switchable main/fallback model" above for the model-level fallback that *is* now configurable.
- **`decision_engine.py` target-selection bug, found while building the real response executor** - every capability, including `block_source`/`rate_limit` (which act on the attacker), was handed `alert.target_asset` (the protected device) as `CyberAction.target`, never `alert.source_id` (the attacker). Harmless while only `SimulatedExecutor` ever read `target` (it only ever logged it); a real bug once `ProductionExecutor` needs to know what to actually block. Fixed and tested - see "Real response execution" above.

**Not fixed - documented, not silently worked around:**
- **No separately-pinned certification trust root.** `integrity/certification.py::verify_release`'s Ed25519 public key lives in the same `artifacts/` tree as the model/manifest/signature it verifies - there is no independent distribution of the trust root. Real PKI trust-root distribution (out-of-band key pinning, a separate distribution channel) is a substantially larger change than Phase-I's scope; this is recorded as a known limitation rather than a silent fix that would overclaim what actually changed.
- **No asset-name-to-IP/CIDR inventory.** `isolate_segment`'s target is an asset *name* (e.g. `"plc-01"`), never a network address, anywhere in this project's data model - `ProductionExecutor` refuses cleanly rather than fabricate a mapping (see "Real response execution" above). A real deployment needs its own asset inventory to make this capability's real backend actionable.
- **Modbus traffic generalization gap**, found while building live attack-scenario streaming: the certified model recognizes real Modbus captures as `Normal` only ~21% of the time (measured, n=100), vs 100% for every other Normal-traffic sensor type in this dataset (see "Live attack-scenario streaming" above). Kept visible via a dedicated diagnostic scenario rather than routed around.
- **`shap`/`lime` are imported by `analytics/explainability.py` but not declared in any `pyproject.toml` optional-dependency group** - a real packaging gap (installing succeeds today only because a prior `pip install` in this environment happened to pull them in already), low severity, not fixed in this pass since it's outside this pass's actual scope.

**Not yet audited:** the end-to-end failure-mode trace (ingestion → feature extraction → model loading → FTIL aggregation → LLM decision engine → capability execution, stage by stage) was scoped but did not complete before the audit pass that found the items above hit a session usage limit.

## What is implemented now

**Training plane**

- dataset schema inspection and leakage warnings;
- mixed numeric/categorical preprocessing;
- train-only fitting of encoders/scalers;
- CNN-BiLSTM model for `[batch, sequence, features]` input;
- centralized training/evaluation;
- Dirichlet non-IID client partitioning;
- Flower `ClientApp`/`ServerApp` (simulation and real deployment-mode TLS, see above);
- structural update validation;
- FedDMC-inspired PCA/clustering/EMA detector (explicitly not an exact reproduction);
- weighted mean, coordinate median, and trimmed-mean aggregation utilities;
- a certification state machine: `status: "trained-uncertified" -> "certified"` only via `scripts/certify_model.py --sign`, SHA-256 manifest + optional Ed25519 signature, `certification.verify_release()` as the single shared check used by both the CLI and the streaming loader.

**Runtime plane**

- Threat Context Generator (`runtime/threat_context.py`) mapping all 15 real model output classes to severity/priority/playbooks (`configs/threat_context.yaml`), with `strict`/`known_classes` warnings so an unmapped class is loud, not silent;
- Agentic Decision Engine (`agentic/decision_engine.py`) turning a threat context into a bounded `CyberActionPlan`, with per-entity interaction history (`agentic/history.py`) - plus a config-selectable LLM-backed alternative (`agentic/llm_engine.py`, LangGraph + local Ollama) behind a real on-box benchmark gate, see above;
- Adaptive Semantic Trust Boundary (`trust_boundary/`): preprocessing, deterministic controls, Sentence-BERT semantic risk, context consistency, IsolationForest behavioral trust, dynamic trust scoring (`T = ws*(1-Rs) + wc*Rc + wb*Rb`), autonomy-mode capability enforcement, output protection, memory integrity, and hash-chained tamper-evident audit logging;
- config-switchable capability execution (`trust_boundary/capability_enforcement.py`): `SimulatedExecutor` (default, logs "would execute") or `ProductionExecutor` (real OS-level actions - Linux `nftables`/`iptables`, Windows `netsh advfirewall`), selected via `trust_boundary.executor.mode`, see "Real response execution" above;
- analyst feedback loop (`analytics/trust_labels.py`, `analytics/threshold_validation.py`): human-labeled ground truth on past trust decisions, referenced by audit-log sequence, hash-chained - see "Analyst feedback loop" above;
- Phase II analytics (`analytics/`): trust forecasting, agent reputation, Page-Hinkley concept-drift detection, SHAP/LIME explainability, KPI aggregation, and a feedback loop, plus `scripts/generate_security_dashboard.py`;
- Streaming feature pipeline (`streaming/`): `CsvReplaySource` -> `StreamingFeatureExtractor` -> `StreamingIDS`, verifying the certified model (`verify_release`) before loading, closing the full loop end to end via `scripts/run_streaming_demo.py`: replay -> IDS inference -> Threat Context Generator -> Agentic Decision Engine -> Trust Boundary -> gated action execution -> audit log;
- live PCAP-paced multi-source streaming and attack-scenario simulation (`streaming/live_source.py`, `streaming/scenario.py`, `scripts/run_attack_scenario.py`), see "Live attack-scenario streaming" above;
- deployment-mode TLS + SuperNode auth (`security/certificates.py`, see above);
- tests across the model, integrity, trust-boundary, analytics, certification, and streaming packages.

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
│   ├── runtime/
│   ├── agentic/
│   ├── trust_boundary/
│   ├── analytics/
│   ├── security/
│   └── streaming/
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
- No live MQTT/Modbus/OPC-UA/SCADA broker exists in this project. `streaming/sources.py::CsvReplaySource` replays a static Edge-IIoTset capture through the same `RecordSource` protocol a real broker client would implement — the seam is deliberately documented, not hidden as if it were live traffic. `scripts/run_streaming_demo.py` defaults to replaying the exact held-out rows Gate 2/3 already scored offline (`heldout_indices`), so any deviation from the offline macro-F1 indicates a pipeline bug, not generalization; a `--all-rows` flag exists for arbitrary replay and prints a loud caveat that any accuracy shown then is not a generalization metric.
- The diagram's "Flow Generation" / windowing box is deliberately **refused**, not stubbed: the Edge-IIoTset CSV is grouped into contiguous per-class blocks (confirmed by direct row inspection), so any `sequence_length > 1` window on this file would leak the label through row position. `streaming/pipeline.py::StreamingIDS` raises `NotImplementedError` on `sequence_length != 1` citing this finding rather than silently producing a temporally-flavored result the data cannot support.
