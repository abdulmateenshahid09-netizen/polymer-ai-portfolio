"""Tests for the report facts. Small examples are calculated by hand.

Hand example (Line 1, ideal rate 2 per minute):
  run A: planned 480, run 400, total 700, good 665
  run B: planned 480, run 440, total 800, good 760
  Totals: planned 960, run 840, total 1500, good 1425; ideal output 2*400 + 2*440 = 1680
  Availability = 840 / 960 = 0.875      Performance = 1500 / 1680 = 0.892857
  Quality      = 1425 / 1500 = 0.95     OEE = 0.875 * 0.892857 * 0.95 = 0.742188
  Losses (minutes): speed 840 - 1500/2 = 90, breakdown 80, changeover 40, scrap 75/2 = 37.5
  Scrap = 75 pieces = 5.0 % of 1500; scratches 45 (60 %), ovality 30 (40 %)
"""
import pandas as pd
import pytest

import simulator
from ai.report_facts import build_facts

T0 = pd.Timestamp("2026-01-01 06:00")


def frames():
    prod = pd.DataFrame({
        "start_time": [T0, T0 + pd.Timedelta(hours=8), T0], "line": ["Line 1", "Line 1", "Line 2"],
        "planned_minutes": [480, 480, 480], "run_minutes": [400, 440, 100],
        "ideal_rate_per_min": [2.0, 2.0, 2.0], "total_count": [700, 800, 50],
        "good_count": [665, 760, 10]})
    down = pd.DataFrame({
        "start_time": [T0, T0, T0], "line": ["Line 1", "Line 1", "Line 2"],
        "category": ["Breakdown", "Changeover", "Breakdown"], "minutes": [80, 40, 999]})
    defects = pd.DataFrame({
        "timestamp": [T0, T0, T0], "line": ["Line 1", "Line 1", "Line 2"],
        "defect_type": ["Scratches", "Ovality", "Specks"], "quantity": [45, 30, 40]})
    return prod, down, defects


def empty(columns):
    return pd.DataFrame({c: [] for c in columns})


def test_oee_loss_and_scrap_by_hand():
    prod, down, defects = frames()
    f = build_facts("Line 1", "All data", None, None, empty(["characteristic"]), empty([]),
                    prod, down, defects)["production"]
    assert f["oee_percent"] == 74.2
    assert f["availability_percent"] == 87.5 and f["quality_percent"] == 95.0
    assert f["performance_percent"] == 89.3
    assert f["biggest_loss"] == "Speed loss (below ideal rate)"
    assert f["biggest_loss_percent_of_planned"] == pytest.approx(9.38, abs=0.01)
    assert f["scrap_percent"] == 5.0
    assert f["top_defects"] == [{"type": "Scratches", "percent_of_scrap": 60.0},
                                {"type": "Ovality", "percent_of_scrap": 40.0}]


def test_other_line_and_old_records_are_left_out():
    prod, down, defects = frames()
    # Line 2 would give OEE 6 %, scrap 80 %: none of it may leak into Line 1
    f = build_facts("Line 1", "Late", T0 + pd.Timedelta(hours=8), None, empty(["characteristic"]),
                    empty([]), prod, down, defects)
    assert f["production"]["planned_minutes"] == 480.0       # only run B is inside the period
    assert f["production"]["scrap_percent"] == 0.0 and f["production"]["top_defects"] == []


def test_missing_data_gives_notes_not_errors():
    prod, down, defects = frames()
    f = build_facts("Line 3", "All data", None, None, empty(["characteristic"]), empty([]),
                    prod, down, defects)
    assert f["wall_thickness"] is None and f["production"] is None
    assert len(f["notes"]) == 2


@pytest.fixture(scope="module")
def demo():
    meas, _proc, prod, down, defects, _events = simulator.build_tables()
    specs = pd.DataFrame({"product": ["PE100 SDR11 110mm"], "characteristic": ["wall_thickness"],
                          "lsl": [10.0], "usl": [11.1], "target": [10.4]})
    meas["timestamp"] = pd.to_datetime(meas["timestamp"])
    for df in (prod, down):
        df["start_time"] = pd.to_datetime(df["start_time"])
    defects["timestamp"] = pd.to_datetime(defects["timestamp"])
    end = max(meas["timestamp"].max(), pd.to_datetime(prod["end_time"]).max())
    return meas, specs, prod, down, defects, end


def facts(demo, line, days):
    meas, specs, prod,down, defects, end = demo
    start = None if days is None else end - pd.Timedelta(days=days)
    return build_facts(line, "test", start, end, meas, specs, prod, down, defects)


def test_all_data_matches_the_dashboard(demo):
    one, two = facts(demo, "Line 1", None), facts(demo, "Line 2", None)
    q = one["wall_thickness"]
    assert (q["cpk"], q["ppk"], q["spc_alarm_events"], q["ai_anomaly_events"]) == (1.739, 1.089, 19, 16)
    q = two["wall_thickness"]
    assert (q["cpk"], q["ppk"], q["spc_alarm_events"], q["ai_anomaly_events"]) == (1.57, 1.008, 31, 11)
    assert (one["production"]["oee_percent"], two["production"]["oee_percent"]) == (75.1, 72.9)


def test_last_7_days_use_the_baseline_of_the_whole_history(demo):
    # the AI insights page shows 7 / 4 for Line 1 and 3 / 1 for Line 2 in the last 7 days;
    # a baseline taken from the last 7 days only gives different counts
    a, b = facts(demo, "Line 1", 7)["wall_thickness"], facts(demo, "Line 2", 7)["wall_thickness"]
    assert (a["readings"], a["spc_alarm_events"], a["ai_anomaly_events"]) == (336, 7, 4)
    assert (b["spc_alarm_events"], b["ai_anomaly_events"]) == (3, 1)


def test_too_few_readings_gives_no_alarm_counts(demo):
    meas, specs, prod, down, defects, end = demo
    few = meas[(meas["line"] == "Line 1") & (meas["characteristic"] == "wall_thickness")].head(100)
    f = build_facts("Line 1", "x", None, end, few, specs, prod, down, defects)
    assert "spc_alarm_events" not in f["wall_thickness"] and "cpk" in f["wall_thickness"]
    assert any("alarm" in n for n in f["notes"])