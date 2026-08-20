"""CLI entry point for Concept Drift Analytics (analytics/drift.py) - previously
tested but never invoked outside tests/test_drift.py, per an architecture audit.

Two genuinely independent data sources/instantiations (see drift.py's own
docstrings for the full distinction, restated briefly here): audit_log_drift
reads the Trust Boundary's agentic-side audit log (a generic score-shift
signal, no attack-detection claim); ftil_trust_drift reads the FTIL-side
per-round client trust log (the only one structurally relevant to a
gradual_scaling-style attack). Both are reported here since they answer
different questions, not merged into one number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tfacd.analytics.drift import audit_log_drift, ftil_trust_drift

parser = argparse.ArgumentParser()
parser.add_argument("--audit-log", default="artifacts/trust_boundary/audit_log.jsonl")
parser.add_argument("--ftil-trust-log", default="artifacts/models/ftil_trust_log.jsonl")
parser.add_argument("--delta", type=float, default=0.005)
parser.add_argument("--lam", type=float, default=1.0)
args = parser.parse_args()

if Path(args.audit_log).exists():
    print(f"=== Trust Boundary audit log drift ({args.audit_log}) ===")
    result = audit_log_drift(args.audit_log, delta=args.delta, lam=args.lam)
    if not result:
        print("  (no scored entries)")
    for agent_id, fields in sorted(result.items()):
        fired = {f: points for f, points in fields.items() if points}
        if fired:
            print(f"  {agent_id:<22} drift detected: {fired}")
        else:
            print(f"  {agent_id:<22} no drift detected")
else:
    print(f"=== Trust Boundary audit log drift: SKIPPED ({args.audit_log} not found - run scripts/run_trust_boundary_demo.py first) ===")

print()

if Path(args.ftil_trust_log).exists():
    print(f"=== FTIL trust log drift ({args.ftil_trust_log}) - the only one relevant to a gradual_scaling-style attack ===")
    result = ftil_trust_drift(args.ftil_trust_log, delta=args.delta, lam=args.lam)
    if not result:
        print("  (no client trust scores)")
    for client_id, points in sorted(result.items()):
        if points:
            print(f"  client {client_id:<5} drift detected at round index(es): {points}")
        else:
            print(f"  client {client_id:<5} no drift detected")
else:
    print(f"=== FTIL trust log drift: SKIPPED ({args.ftil_trust_log} not found - run `flwr run . --stream` with use-ftil=true first) ===")

print("\nStructural argument, not a benchmarked result: no experiment here measures detection lag/precision against a live attack. See drift.py's module docstring.")
