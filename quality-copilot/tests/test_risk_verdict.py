"""Tests for the plain-language verdict that compares the risk model with the simple rules."""
import pandas as pd

from ai.risk_model import compare_verdict


def _table(model, naive_a, naive_b):
    rows = []
    for name, (auc, ap) in [("XGBoost model", model), ("naive |z|", naive_a),
                            ("naive |rolling mean|", naive_b)]:
        rows.append({"method": name, "AUC": auc, "avg precision": ap})
    return pd.DataFrame(rows)


def test_model_clearly_better_on_both_measures_is_reported_as_better():
    beats, text = compare_verdict(_table((0.80, 0.50), (0.70, 0.40), (0.60, 0.30)))
    assert beats is True
    assert "beats" in text and "NOT" not in text


def test_model_equal_to_the_best_rule_is_not_reported_as_better():
    beats, text = compare_verdict(_table((0.65, 0.24), (0.65, 0.33), (0.60, 0.23)))
    assert beats is False
    assert "does NOT clearly beat" in text
    assert "0.65" in text and "0.33" in text            # the best simple rule is named


def test_better_auc_alone_is_not_enough():
    beats, _ = compare_verdict(_table((0.80, 0.30), (0.70, 0.40), (0.60, 0.30)))
    assert beats is False


def test_margin_is_respected():
    assert compare_verdict(_table((0.71, 0.41), (0.70, 0.40), (0.60, 0.30)))[0] is False
    assert compare_verdict(_table((0.73, 0.43), (0.70, 0.40), (0.60, 0.30)))[0] is True


def test_one_class_test_period_cannot_be_judged():
    beats, text = compare_verdict(_table((float("nan"), float("nan")), (float("nan"), float("nan")),
                                         (float("nan"), float("nan"))))
    assert beats is None and "cannot be judged" in text
