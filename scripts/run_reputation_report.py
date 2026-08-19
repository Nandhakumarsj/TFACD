"""CLI entry point for Cross-Agent Reputation (analytics/reputation.py) -
previously tested but never invoked outside tests/test_reputation.py, per an
architecture audit. Reads a real trust-boundary audit log and prints agents
ranked best to worst.
"""

from __future__ import annotations

import argparse

from tfacd.analytics.reputation import rank_agents

parser = argparse.ArgumentParser()
parser.add_argument("--audit-log", default="artifacts/trust_boundary/audit_log.jsonl")
parser.add_argument("--recent-window", type=int, default=5, help="mean of the last N scored decisions per agent")
parser.add_argument("--violation-penalty", type=float, default=0.15)
args = parser.parse_args()

rankings = rank_agents(args.audit_log, recent_window=args.recent_window, violation_penalty=args.violation_penalty)

print(f"{'rank':>4} {'agent_id':<22} {'reputation':>10} {'interactions':>12} {'violations':>10}")
for rank, r in enumerate(rankings, start=1):
    print(f"{rank:>4} {r.agent_id:<22} {r.reputation_score:>10.3f} {r.num_interactions:>12} {r.num_violations:>10}")

print(f"\n{len(rankings)} agent(s) ranked from {args.audit_log}")
print("reputation_score is NOT clamped to [0,1] like TrustScores - repeated violations can push it negative.")
