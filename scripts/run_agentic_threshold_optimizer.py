"""Report ASTB trust-weight/threshold candidates from analyst feedback.

This is distinct from run_threshold_optimizer.py, which tunes the FTIL client
update detector. This command never writes configs or changes live thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfacd.analytics.feedback_loop import AnalystFeedbackStore, run_agentic_grid_search


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--feedback", default="artifacts/analytics/analyst_feedback.jsonl")
parser.add_argument("--output", default="artifacts/analytics/agentic_threshold_report.json")
args = parser.parse_args()

records = AnalystFeedbackStore(args.feedback).load_feedback()
results = run_agentic_grid_search(records)
payload = {
    "feedback_path": args.feedback,
    "feedback_records": len(records),
    "synthetic_fixture": any("synthetic" in record.notes.lower() for record in records),
    "live_config_changed": False,
    "candidates": [
        {
            "weights": result.weights,
            "thresholds": result.thresholds,
            "f1_score": result.f1_score,
            "precision": result.precision,
            "recall": result.recall,
        }
        for result in results
    ],
}
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print(f"feedback_records={len(records)} candidates={len(results)}")
if results:
    print(f"best_weights={results[0].weights} best_thresholds={results[0].thresholds} f1={results[0].f1_score:.3f}")
else:
    print("No feedback records: no ASTB candidate was selected.")
print(f"Saved: {output_path.resolve()}")