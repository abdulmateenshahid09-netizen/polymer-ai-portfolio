import numpy as np
import pandas as pd
import pytest

from ai.anomaly import build_features, detect_anomalies, evaluate_detection


def _series(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return 10.4 + rng.normal(0, 0.05, n)


def test_features_are_in_baseline_units():
    base = np.array([10.0, 10.2, 10.4, 10.6, 10.8] * 10)
    f = build_features([10.4, 10.4 + base.std(ddof=1)], base)
    assert f["z"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert f["z"].iloc[1] == pytest.approx(1.0, abs=1e-9)
    assert f["diff"].iloc[1] == pytest.approx(1.0, abs=1e-9)


def test_too_short_baseline_is_rejected():
    with pytest.raises(ValueError):
        build_features([1, 2, 3], [1, 2, 3])


def test_identical_baseline_is_rejected():
    with pytest.raises(ValueError):
        build_features([1.0] * 5, [1.0] * 40)


def test_level_shift_is_detected_quickly():
    x = _series()
    x[500:530] += 0.25          # 5 sigma shift
    r = detect_anomalies(x, baseline=x[:300])
    assert r["anomaly"].iloc[500:505].any()
    assert r["anomaly"].iloc[:300].mean() <= 0.03


def test_spread_increase_is_detected():
    x = _series()
    rng = np.random.default_rng(5)
    x[500:530] += rng.normal(0, 0.2, 30)    # same mean, 4x the spread
    r = detect_anomalies(x, baseline=x[:300])
    assert r["anomaly"].iloc[500:535].sum() >= 3


def test_quiet_data_gives_few_flags():
    x = _series()
    r = detect_anomalies(x, baseline=x[:300])
    assert r["anomaly"].iloc[300:].mean() < 0.05


def test_same_seed_same_result():
    x = _series()
    a = detect_anomalies(x, baseline=x[:300], seed=3)["anomaly"]
    b = detect_anomalies(x, baseline=x[:300], seed=3)["anomaly"]
    assert a.equals(b)


def test_evaluation_hand_example():
    ts = pd.date_range("2026-01-01 00:00", periods=96, freq="30min")   # 2 days
    flags = np.zeros(96, dtype=bool)
    flags[10] = True            # false alarm (outside the fault window)
    flags[41] = True            # inside the fault: 1 hour after fault start
    faults = [(ts[40], ts[46])]
    r = evaluate_detection(flags, ts, faults)
    assert r["delays_min"] == [30]
    assert r["detected"] == 1
    assert r["false_alarm_events"] == 1
    # outside readings = 96 - (6 fault + 4 grace) = 86 -> 86/48 days
    assert r["false_alarms_per_day"] == pytest.approx(1 / (86 / 48), abs=0.01)


def test_missed_fault_reported_as_none():
    ts = pd.date_range("2026-01-01", periods=48, freq="30min")
    r = evaluate_detection(np.zeros(48, dtype=bool), ts, [(ts[10], ts[20])])
    assert r["delays_min"] == [None]
    assert r["detected"] == 0
