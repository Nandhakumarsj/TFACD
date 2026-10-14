"""Apply the small in-place changes required by the second-review benchmark suite.

Run from the repository root after copying/unzipping the extension bundle into it:

    python scripts/apply_second_review_extension.py

The script is idempotent and refuses to modify a file when an expected source
pattern is missing, so a changed local codebase is not silently rewritten.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"[already applied] {label}")
        return False
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one source match in {path}, found {count}; inspect local changes before applying.")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"[applied] {label}")
    return True


# 1) Leakage-safe session-temporal splitting for the existing sequence_length=1 runtime.
pre = ROOT / "src" / "tfacd" / "data" / "preprocess.py"
replace_once(
    pre,
    '''def _split_temporal_ids_by_session(\n    seq_meta: pd.DataFrame,\n    test_size: float,\n    val_size: float,\n    seed: int = 42,\n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n    """Split temporal windows at the session boundary to avoid train/test leakage.\n    Windows from the same session stay together in exactly one split.\n    """\n    session_ids = seq_meta["session_id"].drop_duplicates().to_numpy()\n    if len(session_ids) == 0:\n        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)\n\n    rng = np.random.default_rng(seed)\n    shuffled = session_ids.copy()\n    rng.shuffle(shuffled)\n''',
    '''def _split_temporal_ids_by_session(\n    seq_meta: pd.DataFrame,\n    test_size: float,\n    val_size: float,\n    seed: int = 42,\n    chronological: bool = False,\n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n    """Split windows by whole sessions, optionally in chronological order."""\n    session_ids = seq_meta["session_id"].drop_duplicates().to_numpy()\n    if len(session_ids) == 0:\n        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)\n\n    if chronological:\n        ordered = (\n            seq_meta.groupby("session_id", as_index=False)\n            .agg(start_time=("start_time", "min"), end_time=("end_time", "max"))\n            .sort_values(["end_time", "start_time", "session_id"], kind="mergesort")\n        )\n        shuffled = ordered["session_id"].to_numpy()\n    else:\n        rng = np.random.default_rng(seed)\n        shuffled = session_ids.copy()\n        rng.shuffle(shuffled)\n''',
    "preprocess: add chronological session split mode",
)

replace_once(
    pre,
    '''    seed = int(config.get("seed", 42))\n    test_size = float(cfg.get("test_size", 0.15))\n    val_size = float(cfg.get("validation_size", 0.15))\n\n    idx = np.arange(len(y))\n    idx_train_val, idx_test, y_train_val, y_test = train_test_split(\n        idx, y, test_size=test_size, random_state=seed, stratify=y\n    )\n    relative_val = val_size / (1.0 - test_size)\n    idx_train, idx_val, y_train, y_val = train_test_split(\n        idx_train_val,\n        y_train_val,\n        test_size=relative_val,\n        random_state=seed,\n        stratify=y_train_val,\n    )\n    x_train, x_val, x_test = x.iloc[idx_train], x.iloc[idx_val], x.iloc[idx_test]\n''',
    '''    seed = int(config.get("seed", 42))\n    test_size = float(cfg.get("test_size", 0.15))\n    val_size = float(cfg.get("validation_size", 0.15))\n    split_strategy = cfg.get("split_strategy", "row_random")\n\n    if split_strategy == "session_temporal":\n        temporal_cfg = cfg.get("temporal", {})\n        timestamp_column = temporal_cfg.get("timestamp_column", cfg.get("timestamp_column", "auto"))\n        if timestamp_column == "auto":\n            timestamp_column = _find_candidate(list(frame.columns), TIME_CANDIDATES)\n        if not timestamp_column:\n            raise ValueError("data.split_strategy='session_temporal' requires a timestamp column")\n        seq_meta, window_indices = build_sequence_index(\n            frame, sequence_length=1, stride=1, timestamp_column=timestamp_column,\n            group_columns=temporal_cfg.get("group_columns"),\n            inactivity_seconds=temporal_cfg.get("inactivity_seconds", 30.0),\n            label_column=target_col,\n        )\n        train_ids, val_ids, test_ids = _split_temporal_ids_by_session(\n            seq_meta, test_size=test_size, val_size=val_size, seed=seed, chronological=True\n        )\n        if len(train_ids) == 0 or len(val_ids) == 0 or len(test_ids) == 0:\n            raise ValueError(\n                "Session-safe temporal row split failed: each of train/validation/test must contain at least one session. "\n                f"Found train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)} windows across "\n                f"{seq_meta['session_id'].nunique()} session(s)."\n            )\n        idx_train = window_indices[train_ids, 0]\n        idx_val = window_indices[val_ids, 0]\n        idx_test = window_indices[test_ids, 0]\n        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]\n        split_metadata = {\n            "split_strategy": "session_temporal",\n            "timestamp_column": timestamp_column,\n            "inactivity_seconds": temporal_cfg.get("inactivity_seconds", 30.0),\n            "sessions_train": int(seq_meta.iloc[train_ids]["session_id"].nunique()),\n            "sessions_val": int(seq_meta.iloc[val_ids]["session_id"].nunique()),\n            "sessions_test": int(seq_meta.iloc[test_ids]["session_id"].nunique()),\n        }\n    elif split_strategy == "row_random":\n        idx = np.arange(len(y))\n        idx_train_val, idx_test, y_train_val, y_test = train_test_split(\n            idx, y, test_size=test_size, random_state=seed, stratify=y\n        )\n        relative_val = val_size / (1.0 - test_size)\n        idx_train, idx_val, y_train, y_val = train_test_split(\n            idx_train_val, y_train_val, test_size=relative_val, random_state=seed, stratify=y_train_val\n        )\n        split_metadata = {"split_strategy": "row_random"}\n    else:\n        raise ValueError(f"unknown data.split_strategy={split_strategy!r}; expected 'row_random' or 'session_temporal'")\n\n    x_train, x_val, x_test = x.iloc[idx_train], x.iloc[idx_val], x.iloc[idx_test]\n''',
    "preprocess: make row-based preprocessing leakage-safe",
)

replace_once(
    pre,
    '    metadata = {\n        "split_mode": "row",\n        "target_column": target_col,\n',
    '    metadata = {\n        "split_mode": "row",\n        **split_metadata,\n        "target_column": target_col,\n',
    "preprocess: persist split strategy metadata",
)

# 2) Chronological option for the true sequence-window branch.
replace_once(
    pre,
    '    train_ids, val_ids, test_ids = _split_temporal_ids_by_session(seq_meta, test_size, val_size, seed=seed)\n',
    '    train_ids, val_ids, test_ids = _split_temporal_ids_by_session(\n        seq_meta, test_size, val_size, seed=seed,\n        chronological=bool(temporal_cfg.get("chronological_split", False)),\n    )\n',
    "preprocess: expose chronological temporal-window split",
)

# 3) Second-review config: keep one-step model input, change only the splitting strategy.
config_path = ROOT / "configs" / "edge_iiot.yaml"
replace_once(
    config_path,
    '  test_size: 0.15\n  validation_size: 0.15\n  max_rows: null\n',
    '  test_size: 0.15\n  validation_size: 0.15\n  split_strategy: "session_temporal"\n  max_rows: null\n',
    "config: enable session-temporal row split",
)
replace_once(
    config_path,
    '    inactivity_seconds: 30\n    group_columns: null\n    # set null for full file audit (computationally expensive).\n',
    '    inactivity_seconds: 30\n    group_columns: null\n    chronological_split: true\n    # set null for full file audit (computationally expensive).\n',
    "config: enable chronological ordering",
)
replace_once(
    config_path,
    '  rounds: 3\n  local_epochs: 1\n',
    '  rounds: 3\n  local_epochs: 1\n  security_experiment_proximal_mu: 0.01\n',
    "config: record FedProx benchmark mu",
)

# 4) Partition utility accepts separate experiment partition sets without changing the production default.
part = ROOT / "scripts" / "create_partitions.py"
replace_once(
    part,
    'parser = argparse.ArgumentParser()\nparser.add_argument("--config", default="configs/edge_iiot.yaml")\nargs = parser.parse_args()\n',
    'parser = argparse.ArgumentParser()\nparser.add_argument("--config", default="configs/edge_iiot.yaml")\nparser.add_argument("--num-clients", type=int, default=None, help="override federated.num_clients for an experiment partition set")\nparser.add_argument("--output-dir", default=None, help="override partition output directory")\nargs = parser.parse_args()\n',
    "partitions: add experiment overrides",
)
replace_once(
    part,
    'fed = config["federated"]\nif fed["partition_mode"] == "iid":\n    partitions = iid_partition(prepared.y_train, int(fed["num_clients"]), int(config["seed"]))\n',
    'fed = config["federated"]\nnum_clients = int(args.num_clients if args.num_clients is not None else fed["num_clients"])\nif fed["partition_mode"] == "iid":\n    partitions = iid_partition(prepared.y_train, num_clients, int(config["seed"]))\n',
    "partitions: use override client count",
)
replace_once(part, '        int(fed["num_clients"]),\n', '        num_clients,\n', "partitions: use override in Dirichlet mode")
replace_once(part, 'out = Path(config["data"]["output_dir"]) / "partitions"\n', 'out = Path(args.output_dir) if args.output_dir else Path(config["data"]["output_dir"]) / "partitions"\n', "partitions: use override output directory")

# 5) Restore deterministic test doubles for tests that explicitly exercise autonomous execution.
boundary_test = ROOT / "tests" / "test_boundary.py"
if boundary_test.exists():
    replace_once(boundary_test,
        "from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, SessionContext, ThreatContext\n",
        "from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, SessionContext, ThreatContext, TrustScores\n",
        "test_boundary: import TrustScores for verified test regulator")
    replace_once(boundary_test,
        "def build_boundary(tmp_path, executor=None):\n    return AdaptiveSemanticTrustBoundary(\n",
        "class AlwaysVerifiedRegulator:\n    \"\"\"Deterministic test double for tests that specifically exercise autonomous execution.\"\"\"\n    def evaluate(self, semantic_risk, context_consistency, behavioral_trust):\n        return TrustScores(semantic_risk=semantic_risk, context_consistency=context_consistency, behavioral_trust=behavioral_trust, trust_value=1.0)\n    def trust_level(self, trust_value):\n        return \"verified\"\n    def autonomy_mode(self, trust_level):\n        return \"autonomous_execution\"\n\n\ndef build_boundary(tmp_path, executor=None, trust_regulator=None):\n    return AdaptiveSemanticTrustBoundary(\n",
        "test_boundary: add verified regulator")
    replace_once(boundary_test,
        "        trust_regulator=DynamicTrustScoreRegulator(0.4, 0.3, 0.3, THRESHOLDS),\n",
        "        trust_regulator=trust_regulator or DynamicTrustScoreRegulator(0.4, 0.3, 0.3, THRESHOLDS),\n",
        "test_boundary: allow regulator injection")
    replace_once(boundary_test,
        "    decision = build_boundary(tmp_path).evaluate(plan, context, make_session(\"agent-happy\"))\n",
        "    decision = build_boundary(tmp_path, trust_regulator=AlwaysVerifiedRegulator()).evaluate(plan, context, make_session(\"agent-happy\"))\n",
        "test_boundary: make happy path deterministic")
    replace_once(boundary_test,
        "    decision = build_boundary(tmp_path, executor=FailsSpecificCapabilityExecutor()).evaluate(plan, context, make_session(\"agent-exec-partial\"))\n",
        "    decision = build_boundary(tmp_path, executor=FailsSpecificCapabilityExecutor(), trust_regulator=AlwaysVerifiedRegulator()).evaluate(plan, context, make_session(\"agent-exec-partial\"))\n",
        "test_boundary: make partial execution deterministic")

# 6) Keep the real Ollama integration audit isolated per test run.
llm_test = ROOT / "tests" / "test_llm_engine_ollama_integration.py"
if llm_test.exists():
    replace_once(llm_test,
        "def test_real_ollama_plan_passes_the_full_trust_boundary():\n",
        "def test_real_ollama_plan_passes_the_full_trust_boundary(tmp_path):\n",
        "test_ollama: use pytest tmp_path")
    replace_once(llm_test,
        "        audit_logger=AuditLogger(Path(\"artifacts/agentic/ollama_integration_audit.jsonl\")),\n",
        "        audit_logger=AuditLogger(tmp_path / \"ollama_integration_audit.jsonl\"),\n",
        "test_ollama: isolate audit artifact")

# 5) Fix the artifact-gated FTIL drift test instead of commenting it out when runtime node IDs change.
drift = ROOT / "tests" / "test_drift.py"
replace_once(drift, 'from pathlib import Path\n', 'import json\nfrom pathlib import Path\n', "test_drift: import JSON parser")
replace_once(
    drift,
    '''    result = ftil_trust_drift(_FTIL_LOG)\n    assert set(result) == {"0", "1", "2", "3", "4"}\n    for drift_points in result.values():\n''',
    '''    result = ftil_trust_drift(_FTIL_LOG)\n    expected_clients: set[str] = set()\n    for line in _FTIL_LOG.read_text(encoding="utf-8").splitlines():\n        if not line.strip():\n            continue\n        expected_clients.update(json.loads(line).get("trust_scores", {}).keys())\n    assert set(result) == expected_clients\n    for drift_points in result.values():\n''',
    "test_drift: accept Flower runtime node IDs",
)

print("\nSecond-review in-place changes applied.")
print("Next: regenerate artifacts/data/prepared.npz after the session-temporal split change before using new metrics.")
