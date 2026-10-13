"""OEE tests. Every expected number below was calculated by hand.

Two shifts, ideal rate 1 per minute:
  shift 1: planned 480, run 400, total 360, good 342
  shift 2: planned 480, run 440, total 400, good 400
Totals: planned 960, run 840, total 760, good 742
  Availability = 840 / 960 = 0.875
  Performance  = 760 / 840 = 0.904762
  Quality      = 742 / 760 = 0.976316
  OEE          = 0.875 * 0.904762 * 0.976316 = 742 / 960 = 0.772917
"""
import pandas as pd
import pytest

from metrics.oee import calculate_oee, loss_breakdown, oee_by

PROD = pd.DataFrame({
    "start_time": ["2026-01-01 06:00:00", "2026-01-01 14:00:00"],
    "line": ["Line 1", "Line 1"], "shift": ["Day", "Evening"],
    "planned_minutes": [480, 480], "run_minutes": [400, 440],
    "ideal_rate_per_min": [1.0, 1.0], "total_count": [360, 400], "good_count": [342, 400],
})
DOWN = pd.DataFrame({"category": ["Changeover", "Breakdown"], "minutes": [60, 40]})


def test_oee_factors():
    r = calculate_oee(PROD)
    assert r["availability"] == pytest.approx(0.875, abs=1e-4)
    assert r["performance"] == pytest.approx(0.9048, abs=1e-4)
    assert r["quality"] == pytest.approx(0.9763, abs=1e-4)
    assert r["oee"] == pytest.approx(0.7729, abs=1e-4)


def test_oee_equals_good_output_over_ideal_output():
    r = calculate_oee(PROD)
    assert r["oee"] == pytest.approx(742 / 960, abs=1e-4)


def test_long_shifts_count_more_than_short_ones():
    # A: 100 planned minutes, OEE 100%.  B: 900 planned minutes, OEE 25%.
    # Correct combined OEE = 325 / 1000 = 0.325. Averaging the two ratios would
    # wrongly give 0.625.
    two = pd.DataFrame({
        "planned_minutes": [100, 900], "run_minutes": [100, 450],
        "ideal_rate_per_min": [1.0, 1.0], "total_count": [100, 450], "good_count": [100, 225],
    })
    assert calculate_oee(two)["oee"] == pytest.approx(0.325, abs=1e-4)


def test_loss_breakdown_adds_up_to_planned_time():
    r = loss_breakdown(PROD, DOWN)
    losses = r["losses"].set_index("loss")["minutes"]
    assert losses["Changeover"] == 60
    assert losses["Breakdown"] == 40
    assert losses["Unclassified downtime"] == pytest.approx(20.0)   # 120 down - 100 classified
    assert losses["Speed loss (below ideal rate)"] == pytest.approx(80.0)  # (400-360)+(440-400)
    assert losses["Scrap and rework"] == pytest.approx(18.0)               # 360 - 342
    assert r["productive_minutes"] == pytest.approx(742.0)                 # 342 + 400
    assert r["losses"]["minutes"].sum() + r["productive_minutes"] == pytest.approx(960.0)
    assert r["productive_percent"] == pytest.approx(77.29, abs=0.01)


def test_losses_are_sorted_largest_first():
    r = loss_breakdown(PROD, DOWN)
    minutes = list(r["losses"]["minutes"])
    assert minutes == sorted(minutes, reverse=True)


def test_oee_by_shift():
    out = oee_by(PROD, "shift").set_index("shift")
    assert out.loc["Day", "oee"] == pytest.approx(342 / 480, abs=1e-4)      # 0.7125
    assert out.loc["Evening", "oee"] == pytest.approx(400 / 480, abs=1e-4)  # 0.8333


def test_performance_above_100_percent_is_flagged():
    fast = PROD.copy()
    fast["ideal_rate_per_min"] = 0.5
    assert any("ideal rate" in w for w in calculate_oee(fast)["warnings"])


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        calculate_oee(PROD.drop(columns=["good_count"]))
    with pytest.raises(ValueError):
        calculate_oee(PROD.iloc[0:0])
