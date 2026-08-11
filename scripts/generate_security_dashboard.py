from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tfacd.analytics.kpi import DEFAULT_AUDIT_LOG, AgentKPISummary, KPIReport, compute_kpis

# Rendering only lives here, not in analytics/kpi.py, so the KPI math stays
# testable without an HTML harness. Self-contained: inline CSS, no external
# CDN/JS, no HTTP server - this repo has zero HTTP services anywhere.

parser = argparse.ArgumentParser()
parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG)
parser.add_argument("--output", default="artifacts/trust_boundary/security_dashboard.html")
parser.add_argument("--top-n", type=int, default=3)
parser.add_argument("--min-scored", type=int, default=2, help="minimum scored entries for an agent to appear in top/bottom rankings")
args = parser.parse_args()


def _pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def _bar(fraction: float, color: str) -> str:
    width = max(0.0, min(1.0, fraction)) * 100
    return f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div></div>'


def _agent_row(summary: AgentKPISummary) -> str:
    trust_cell = f"{summary.mean_trust_value:.3f}{_bar(summary.mean_trust_value, '#3b6fd6')}" if summary.mean_trust_value is not None else "n/a (no scored entries)"
    return (
        "<tr>"
        f"<td>{summary.agent_id}</td>"
        f"<td>{summary.num_interactions}</td>"
        f"<td>{summary.num_scored}</td>"
        f"<td>{_pct(summary.acceptance_rate)}{_bar(summary.acceptance_rate, '#2f9e58')}</td>"
        f"<td>{trust_cell}</td>"
        "</tr>"
    )


def _agent_list_items(summaries: list[AgentKPISummary]) -> str:
    if not summaries:
        return "<li><em>no agent had at least --min-scored scored entries</em></li>"
    return "".join(f"<li>{s.agent_id} - mean trust {s.mean_trust_value:.3f} ({s.num_scored} scored entries)</li>" for s in summaries)


def render_html(report: KPIReport, audit_log_path: str, top_n: int, min_scored: int) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    dist = report.trust_level_distribution
    dist_rows = "".join(
        f"<tr><td>{level}</td><td>{count}</td><td>{_bar(count / report.num_entries if report.num_entries else 0.0, '#8a5fd6')}</td></tr>"
        for level, count in dist.items()
    )
    agent_rows = "".join(_agent_row(s) for s in sorted(report.per_agent, key=lambda s: s.agent_id))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TFACD Trust Boundary Security Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; background: #fff; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0; }}
  .meta {{ color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  .stat-row {{ display: flex; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .stat {{ border: 1px solid #ddd; border-radius: 8px; padding: 0.75rem 1rem; min-width: 160px; }}
  .stat .label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; }}
  .stat .value {{ font-size: 1.6rem; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
  th {{ color: #666; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }}
  .bar-track {{ background: #eee; border-radius: 4px; height: 6px; margin-top: 3px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .rank-cols {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
  .rank-cols > div {{ flex: 1; min-width: 260px; }}
  .caveat {{ color: #666; font-size: 0.8rem; margin-top: -1rem; margin-bottom: 1.5rem; }}
</style>
</head>
<body>
<h1>TFACD Trust Boundary Security Dashboard</h1>
<div class="meta">Generated {generated_at} from {audit_log_path}</div>

<div class="stat-row">
  <div class="stat"><div class="label">Audit entries</div><div class="value">{report.num_entries}</div></div>
  <div class="stat"><div class="label">Overall acceptance rate</div><div class="value">{_pct(report.overall_acceptance_rate)}</div></div>
  <div class="stat"><div class="label">Hard Stage-1/2 rejections</div><div class="value">{dist['hard_rejected']}</div></div>
</div>

<h2>Trust level distribution</h2>
<p class="caveat">"hard_rejected" entries never reached trust scoring (rejected at preprocessing/deterministic_controls) - shown separately, not folded into a trust level.</p>
<table>
  <tr><th>Level</th><th>Count</th><th></th></tr>
  {dist_rows}
</table>

<h2>Per-agent summary</h2>
<table>
  <tr><th>Agent</th><th>Interactions</th><th>Scored</th><th>Acceptance rate</th><th>Mean trust value (scored only)</th></tr>
  {agent_rows}
</table>

<h2>Top {top_n} / bottom {top_n} agents by mean trust value</h2>
<p class="caveat">Agents with fewer than {min_scored} scored entries are excluded - not enough signal for a stable mean.</p>
<div class="rank-cols">
  <div><h3>Top {top_n}</h3><ol>{_agent_list_items(report.top_agents)}</ol></div>
  <div><h3>Bottom {top_n}</h3><ol>{_agent_list_items(report.bottom_agents)}</ol></div>
</div>
</body>
</html>
"""


report = compute_kpis(args.audit_log, top_n=args.top_n, min_scored=args.min_scored)
html = render_html(report, str(args.audit_log), args.top_n, args.min_scored)

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(html, encoding="utf-8")

print(
    f"entries={report.num_entries} acceptance_rate={_pct(report.overall_acceptance_rate)} "
    f"hard_rejected={report.trust_level_distribution['hard_rejected']} agents={len(report.per_agent)}"
)
print(f"Saved: {output_path.resolve()}")
