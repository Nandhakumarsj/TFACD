import pytest

from tfacd.trust_boundary.dynamic_trust import DynamicTrustScoreRegulator

THRESHOLDS = {"low": 0.40, "medium": 0.65, "high": 0.85}


def make_regulator():
    return DynamicTrustScoreRegulator(0.4, 0.3, 0.3, THRESHOLDS)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        DynamicTrustScoreRegulator(0.5, 0.3, 0.3, THRESHOLDS)


def test_trust_value_formula():
    regulator = make_regulator()
    scores = regulator.evaluate(semantic_risk=0.0, context_consistency=1.0, behavioral_trust=1.0)
    assert scores.trust_value == pytest.approx(1.0)

    scores = regulator.evaluate(semantic_risk=1.0, context_consistency=0.0, behavioral_trust=0.0)
    assert scores.trust_value == pytest.approx(0.0)


@pytest.mark.parametrize(
    "value,expected_level",
    [(0.0, "low"), (0.39, "low"), (0.40, "medium"), (0.64, "medium"), (0.65, "high"), (0.84, "high"), (0.85, "verified"), (1.0, "verified")],
)
def test_trust_level_thresholds(value, expected_level):
    regulator = make_regulator()
    assert regulator.trust_level(value) == expected_level


def test_autonomy_mode_mapping():
    regulator = make_regulator()
    assert regulator.autonomy_mode("low") == "read_only"
    assert regulator.autonomy_mode("medium") == "recommendation"
    assert regulator.autonomy_mode("high") == "restricted_action"
    assert regulator.autonomy_mode("verified") == "autonomous_execution"
