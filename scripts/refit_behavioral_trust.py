"""Diagnostic: refits the Behavioral Trust Engine's IsolationForest from real
observed EntityHistory data instead of its synthetic cold-start population,
via BehavioralTrustEngine.refit_from_history(). Deliberately not wired into
the live pipeline (boundary.py never calls this) - an explicit, operator-run
capability, the same posture as run_threshold_optimizer.py for the FL-side
detector.

Honesty note: this script does NOT persist the refit model anywhere - no
pickle/serialization path exists for IsolationForest in this project. It
demonstrates the adaptation's concrete effect (before/after scores on a
handful of representative plans) so an operator can judge whether refitting
would help, not silently change production behavior. Wiring persistence in is
a separate, bigger design decision (how often to refit, where to store it,
whether a live process should hot-swap its forest) intentionally left open.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from tfacd.agentic.history import EntityHistory
from tfacd.common.config import load_config
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, SessionContext
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--history", default="artifacts/agentic/history.jsonl", help="path to a real, persisted EntityHistory JSONL file")
parser.add_argument("--min-samples", type=int, default=20)
args = parser.parse_args()

config = load_config(args.config)
tb_config = config["trust_boundary"]
policy = load_config(config["runtime"]["trust_policy_path"])
high_risk = set(policy["capability_whitelist"]["high_risk"])

history = EntityHistory(persist_path=args.history)
real_events = history.all_events(kind="trust_decision")
print(f"{len(real_events)} real trust_decision event(s) found across {len({e['entity_id'] for e in real_events})} entit(y/ies) in {args.history}")

engine = BehavioralTrustEngine(high_risk_capabilities=high_risk, ema_alpha=tb_config["ema_alpha"])

probes = [
    ("typical low-risk (observe)", CyberActionPlan(incident_id="probe-1", rationale="r", confidence=0.5, actions=[CyberAction(capability="observe")])),
    ("high-risk-heavy (all high-risk capabilities)", CyberActionPlan(incident_id="probe-2", rationale="r", confidence=0.5, actions=[CyberAction(capability=c) for c in sorted(high_risk)])),
]
probe_history = EntityHistory()  # a fresh, empty per-entity history for each probe score - isolates the forest's own effect from EMA/violation-rate noise


def probe_session(label: str) -> SessionContext:
    return SessionContext(agent_id=f"probe-{label}", session_id="probe", issued_at=datetime.now(timezone.utc), nonce=f"probe-{label}")


print("\nbefore refit (synthetic cold-start population):")
before_scores = {}
for label, plan in probes:
    score = engine.score(probe_session(label), plan, probe_history)
    before_scores[label] = score
    print(f"  {label:<45} Rb={score:.3f}")

refit = engine.refit_from_history(history, min_samples=args.min_samples)
if not refit:
    print(f"\nNOT ENOUGH real data to refit (need >= {args.min_samples} trust_decision events, found {len(real_events)}) - population unchanged.")
    print("Run scripts/run_trust_boundary_demo.py and/or scripts/run_streaming_demo.py a few times first to accumulate more real history, then re-run this script.")
else:
    print("\nafter refit (real observed population):")
    for label, plan in probes:
        score = engine.score(probe_session(f"{label}-after"), plan, probe_history)
        delta = score - before_scores[label]
        print(f"  {label:<45} Rb={score:.3f}  (delta {delta:+.3f})")
    print("\nThis reflects what 'normal' looks like in the real history above - not necessarily what SHOULD be normal.")
    print("Not persisted: the live pipeline (boundary.py) is unaffected by this run.")
