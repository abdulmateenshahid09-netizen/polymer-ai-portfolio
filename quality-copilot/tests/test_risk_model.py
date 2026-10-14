import numpy as np
import pandas as pd
import pytest

from ai.risk_model import (FEATURES, build_dataset, naive_scores, score_report,
                           time_split, train_model)


def _frames(values, n_base=40):
    n = len(values)
    ts = pd.date_range("2026-01-01", periods=n, freq="30min").strftime("%Y-%m-%d %H:%M:%S")
    wall = pd.DataFrame({"timestamp": ts, "value": values, "shift": "Day"})
    proc = pd.DataFrame({"timestamp": ts, "screw_rpm": 100.0, "haul_speed_mpm": 1.0,
                         "melt_temp_c": 210.0, "mfr": 0.3})
    return wall, proc


def _normal(n=60):
    # alternating 10.35 / 10.45: every reading is about 1 baseline sigma from the mean,
    # so no normal reading can be more than 2 sigma away (keeps hand examples exact)
    return 10.4 + 0.05 * np.where(np.arange(n) % 2 == 0, -1.0, 1.0)


def test_target_marks_the_readings_before_a_spike():
    x = _normal()
    x[50] = 11.0                       # a huge spike, many sigma
    wall, proc = _frames(x)
    d = build_dataset(wall, proc, baseline_n=40)
    # readings 46..49 have the spike within their next 4 readings
    assert d["y"].iloc[46:50].eq(1.0).all()
    assert d["y"].iloc[45] == 0.0      # spike is 5 readings ahead: outside the horizon
    assert d["y"].iloc[50] == 0.0      # the spike itself is not "future" for itself


def test_last_readings_have_unknown_target():
    wall, proc = _frames(_normal())
    d = build_dataset(wall, proc, baseline_n=40)
    assert d["y"].iloc[-4:].isna().all()
    assert d["y"].iloc[:-4].notna().all()


def test_features_never_use_the_future():
    x = _normal()
    wall, proc = _frames(x)
    a = build_dataset(wall, proc, baseline_n=40)
    x2 = x.copy()
    x2[55:] += 5.0                     # change only the later readings
    wall2, proc2 = _frames(x2)
    b = build_dataset(wall2, proc2, baseline_n=40)
    pd.testing.assert_frame_equal(a[FEATURES].iloc[:55], b[FEATURES].iloc[:55])


def test_too_few_readings_rejected():
    wall, proc = _frames(_normal(30))
    with pytest.raises(ValueError):
        build_dataset(wall, proc, baseline_n=40)


def test_time_split_is_ordered_with_a_gap():
    n = 40 * 48
    ts = pd.date_range("2026-01-01", periods=n, freq="30min").strftime("%Y-%m-%d %H:%M:%S")
    data = pd.DataFrame({"timestamp": ts, "y": 0.0})
    train, test = time_split(data, train_days=20, gap_hours=2)
    assert pd.to_datetime(train["timestamp"]).max() < pd.to_datetime(test["timestamp"]).min()
    gap = pd.to_datetime(test["timestamp"]).min() - pd.to_datetime(train["timestamp"]).max()
    assert gap >= pd.Timedelta(hours=2)


def test_score_report_perfect_and_hand_example():
    y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    perfect = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.8]
    r = score_report(y, perfect, top_share=0.2)
    assert r["auc"] == pytest.approx(1.0)
    assert r["precision_top"] == pytest.approx(1.0)
    assert r["recall_top"] == pytest.approx(1.0)
    # worst case: both positives ranked last -> top 2 contain no positives
    worst = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
    w = score_report(y, worst, top_share=0.2)
    assert w["precision_top"] == 0.0
    assert w["recall_top"] == 0.0
    assert w["base_rate"] == pytest.approx(0.2)


def test_score_report_single_class_gives_nan():
    r = score_report([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(r["auc"])


def test_train_needs_both_classes():
    train = pd.DataFrame({c: [0.0, 1.0] for c in FEATURES})
    train["y"] = [0.0, 0.0]
    with pytest.raises(ValueError):
        train_model(train)


def test_naive_scores_hand_example():
    df = pd.DataFrame({"z": [-2.0, 1.0], "roll_mean": [0.5, -1.5]})
    s = naive_scores(df)
    assert list(s["naive |z|"]) == [2.0, 1.0]
    assert list(s["naive |rolling mean|"]) == [0.5, 1.5]


def test_model_learns_an_obvious_pattern():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURES})
    df["y"] = (df["roll_mean"] > 0.8).astype(float)
    model = train_model(df)
    p = model.predict_proba(df[FEATURES])[:, 1]
    assert score_report(df["y"], p)["auc"] > 0.95
