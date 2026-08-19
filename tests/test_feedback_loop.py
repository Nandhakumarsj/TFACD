from __future__ import annotations

import pytest

from tfacd.analytics.feedback_loop import GridResult, select_best_candidate

# Pure selection-logic tests only - the actual grid search calls run_benchmark
# (real training), which is far too slow for this suite's ~6s convention. See
# scripts/run_threshold_optimizer.py for exercising run_grid_search itself.


def make_result(reject_below_trust, ema_alpha, tpr, fpr):
    return GridResult({"reject_below_trust": reject_below_trust, "ema_alpha": ema_alpha}, {"tpr": tpr, "fpr": fpr})


def test_selects_max_tpr_within_fpr_tolerance():
    results = [
        make_result(0.25, 0.3, tpr=0.5, fpr=0.20),  # fpr exceeds tolerance, excluded
        make_result(0.35, 0.3, tpr=0.4, fpr=0.05),  # within tolerance, lower tpr
        make_result(0.45, 0.3, tpr=0.6, fpr=0.08),  # within tolerance, highest tpr
    ]
    chosen, met_constraint = select_best_candidate(results, fpr_tolerance=0.1)
    assert met_constraint is True
    assert chosen.params == {"reject_below_trust": 0.45, "ema_alpha": 0.3}


def test_falls_back_to_best_tradeoff_when_none_meet_constraint():
    results = [
        make_result(0.25, 0.3, tpr=0.90, fpr=0.50),  # tpr - fpr = 0.40 (best tradeoff)
        make_result(0.35, 0.5, tpr=0.60, fpr=0.30),  # tpr - fpr = 0.30
        make_result(0.45, 0.7, tpr=0.95, fpr=0.60),  # tpr - fpr = 0.35, but higher tpr alone doesn't win
    ]
    chosen, met_constraint = select_best_candidate(results, fpr_tolerance=0.1)
    assert met_constraint is False
    assert chosen.params == {"reject_below_trust": 0.25, "ema_alpha": 0.3}


def test_candidates_with_undefined_metrics_are_skipped_not_selected():
    results = [
        make_result(0.25, 0.3, tpr=None, fpr=None),  # e.g. tp+fn == 0 in benchmark.py's detection_metrics
        make_result(0.35, 0.5, tpr=0.4, fpr=0.05),
    ]
    chosen, met_constraint = select_best_candidate(results, fpr_tolerance=0.1)
    assert met_constraint is True
    assert chosen.params == {"reject_below_trust": 0.35, "ema_alpha": 0.5}


def test_raises_when_no_candidate_has_usable_metrics():
    results = [make_result(0.25, 0.3, tpr=None, fpr=None)]
    with pytest.raises(ValueError):
        select_best_candidate(results)
