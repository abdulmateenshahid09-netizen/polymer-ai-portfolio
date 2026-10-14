import numpy as np
import pandas as pd
import pytest

from ai.recommender import (backtest, ewma_state, fit_process_model, margin_sigmas, output_kgh,
                            predict_thickness, recommend_settings, residuals, thickness_mm)

K, A_MELT, B_MFR = 2.3, 0.004, 0.15


def _synthetic(n=400, seed=0, drift=None):
    """Noise-free data made from the same physics the model assumes."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="30min").strftime("%Y-%m-%d %H:%M:%S")
    rpm = np.full(n, 100.0)
    melt = 210 + rng.normal(0, 2.0, n)
    mfr = 0.30 + rng.normal(0, 0.04, n)
    haul = np.full(n, 1.2)
    haul[n // 2:] = 1.25                      # operator speeds up halfway
    eff = np.ones(n) if drift is None else drift
    out = K * rpm * (1 + A_MELT * (melt - 210)) * (1 + B_MFR * (mfr - 0.30)) * eff
    return pd.DataFrame({"timestamp": ts, "value": thickness_mm(out, haul), "screw_rpm": rpm,
                         "haul_speed_mpm": haul, "melt_temp_c": melt, "mfr": mfr})


def test_mass_balance_hand_example():
    # 10.4 mm wall, 1.2 m/min, 110 mm pipe, density 950 -> 222.586 kg/h (hand calculation)
    assert float(output_kgh(10.4, 1.2)) == pytest.approx(222.586, rel=1e-4)


def test_thickness_and_output_are_inverse():
    for t, v in [(10.0, 1.1), (10.4, 1.2), (11.1, 1.3)]:
        assert float(thickness_mm(output_kgh(t, v), v)) == pytest.approx(t, abs=1e-9)


def test_slower_haul_gives_thicker_wall():
    out = float(output_kgh(10.4, 1.2))
    assert float(thickness_mm(out, 1.1)) > 10.4 > float(thickness_mm(out, 1.3))


def test_fit_recovers_known_physics():
    d = _synthetic()
    m = fit_process_model(d)
    # log-linear fit of a multiplicative law: coefficients close to the true 0.004 and 0.15
    assert m["melt_coef"] == pytest.approx(A_MELT, abs=3e-4)
    assert m["mfr_coef"] == pytest.approx(B_MFR, abs=0.02)
    assert np.exp(m["intercept"]) == pytest.approx(K, rel=0.01)
    assert np.abs(residuals(m, d)).max() < 0.01


def test_too_few_readings_rejected():
    with pytest.raises(ValueError):
        fit_process_model(_synthetic(20))


def test_ewma_state_hand_examples():
    assert ewma_state([0.2]) == pytest.approx(0.2)
    assert ewma_state([0.0, 0.1], alpha=0.5) == pytest.approx(0.05)     # 0.5*0.1 + 0.5*0
    assert ewma_state([]) == 0.0


def test_prediction_reproduces_the_data_when_state_is_exact():
    d = _synthetic()
    m = fit_process_model(d)
    p = predict_thickness(m, 0.0, d["screw_rpm"], d["haul_speed_mpm"], d["melt_temp_c"], d["mfr"])
    assert np.abs(p - d["value"]).max() < 0.05


def test_model_beats_persistence_when_haul_changes():
    n = 400
    drift = 1 + 0.01 * np.sin(np.arange(n) / 60.0)          # slow hidden efficiency drift
    d = _synthetic(n, drift=drift)
    m = fit_process_model(d)
    b = backtest(m, d).dropna()
    step = b[b["haul_changed"]]
    assert len(step) >= 1
    err_model = (step["actual"] - step["model"]).abs().mean()
    err_persist = (step["actual"] - step["persistence"]).abs().mean()
    assert err_model < err_persist / 3


def test_margin_hand_example():
    assert margin_sigmas(10.4, 10.0, 11.1, 0.1) == pytest.approx(4.0)     # 0.4 mm / 0.1 mm
    assert margin_sigmas(11.0, 10.0, 11.1, 0.1) == pytest.approx(1.0)     # upper side is closer


def _setup():
    d = _synthetic()
    m = fit_process_model(d)
    state = 0.04                                  # efficiency 4 % above normal -> wall too thick
    row = d.iloc[-1]
    return m, state, row


def test_recommendation_hits_target_with_haul_only():
    m, state, row = _setup()
    now = float(predict_thickness(m, state, row["screw_rpm"], row["haul_speed_mpm"],
                                  row["melt_temp_c"], row["mfr"]))
    assert now > 10.6                                     # confirms the situation is off target
    r = recommend_settings(m, state, row["melt_temp_c"], row["mfr"], row["screw_rpm"],
                           row["haul_speed_mpm"], 10.4, 10.0, 11.1, 0.09, n_trials=200)
    assert r["rpm"] == row["screw_rpm"]                   # rpm untouched by default
    assert r["predicted_new"] == pytest.approx(10.4, abs=0.03)
    assert r["reached_target"] and not r["at_limit"]
    assert r["haul_change_pct"] > 0                       # too thick -> haul faster
    # analytic answer: the haul speed that gives exactly 10.4 mm at this output
    base = (m["intercept"] + m["melt_coef"] * (row["melt_temp_c"] - m["melt_ref"])
            + m["mfr_coef"] * (row["mfr"] - m["mfr_ref"]))
    out = float(row["screw_rpm"]) * np.exp(base + state)
    exact = out / float(output_kgh(10.4, 1.0))
    assert r["haul"] == pytest.approx(exact, rel=0.005)


def test_recommendation_stops_at_the_change_limit():
    m, state, row = _setup()
    r = recommend_settings(m, 0.20, row["melt_temp_c"], row["mfr"], row["screw_rpm"],
                           row["haul_speed_mpm"], 10.4, 10.0, 11.1, 0.09,
                           max_change_pct=2.0, n_trials=150)
    assert r["at_limit"]
    assert not r["reached_target"]
    assert r["haul_change_pct"] == pytest.approx(2.0, abs=0.05)


def test_same_seed_same_advice():
    m, state, row = _setup()
    args = (m, state, row["melt_temp_c"], row["mfr"], row["screw_rpm"], row["haul_speed_mpm"],
            10.4, 10.0, 11.1, 0.09)
    a = recommend_settings(*args, n_trials=60, seed=3)
    b = recommend_settings(*args, n_trials=60, seed=3)
    assert a["haul"] == b["haul"]


def test_backtest_does_not_peek_at_the_reading_it_predicts():
    d = _synthetic(120, drift=1 + 0.01 * np.sin(np.arange(120) / 20.0))
    m = fit_process_model(d)
    a = backtest(m, d)
    d2 = d.copy()
    d2.loc[100, "value"] += 0.5                 # change one reading
    b = backtest(m, d2)
    assert a["model"].iloc[100] == pytest.approx(b["model"].iloc[100])   # its own prediction is unchanged
    assert a["model"].iloc[101] != pytest.approx(b["model"].iloc[101])   # the next prediction reacts
