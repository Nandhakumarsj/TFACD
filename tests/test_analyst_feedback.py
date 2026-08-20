from tfacd.analytics.feedback_loop import AnalystFeedbackRecord, AnalystFeedbackStore, run_agentic_grid_search


def test_analyst_feedback_store_and_grid_search(tmp_path):
    store_file = tmp_path / "test_feedback.jsonl"
    store = AnalystFeedbackStore(store_path=store_file)

    rec1 = AnalystFeedbackRecord("p1", "agent-1", 0.05, 0.95, 0.90, True, "correct")
    rec2 = AnalystFeedbackRecord("p2", "agent-1", 0.85, 0.20, 0.30, False, "correct")

    store.add_feedback(rec1)
    store.add_feedback(rec2)

    loaded = store.load_feedback()
    assert len(loaded) == 2
    assert loaded[0].incident_id == "p1"
    assert loaded[1].incident_id == "p2"

    grid_results = run_agentic_grid_search(loaded)
    assert len(grid_results) > 0
    best = grid_results[0]
    assert best.f1_score >= 0.0
