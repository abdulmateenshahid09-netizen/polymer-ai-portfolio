"""Defect analysis tests. Every expected number below was calculated by hand."""
import pandas as pd
import pytest

from metrics.defects import pareto, scrap_rate, defect_mix, lot_summary


def make(rows):
    return pd.DataFrame(rows, columns=["defect_type", "quantity"])


def test_pareto_order_and_percentages():
    # A = 50 + 30 = 80, B = 15, C = 5, total 100
    p = pareto(make([("A", 50), ("A", 30), ("B", 15), ("C", 5)]))
    assert list(p["defect_type"]) == ["A", "B", "C"]
    assert list(p["quantity"]) == [80, 15, 5]
    assert list(p["events"]) == [2, 1, 1]
    assert list(p["percent"]) == [80.0, 15.0, 5.0]
    assert list(p["cumulative_percent"]) == [80.0, 95.0, 100.0]


def test_pareto_vital_few_stops_at_80_percent():
    p = pareto(make([("A", 50), ("A", 30), ("B", 15), ("C", 5)]))
    assert list(p["vital_few"]) == [True, False, False]        # A alone reaches 80%
    q = pareto(make([("A", 50), ("B", 30), ("C", 15), ("D", 5)]))
    assert list(q["vital_few"]) == [True, True, False, False]  # A+B = 80%


def test_scrap_rate_by_line():
    prod = pd.DataFrame({"line": ["Line 1", "Line 2"], "total_count": [1000, 500]})
    defects = pd.DataFrame({"line": ["Line 1", "Line 1", "Line 2"],
                            "quantity": [30, 20, 40]})
    r = scrap_rate(defects, prod, "line").set_index("line")
    assert r.loc["Line 1", "scrap"] == 50
    assert r.loc["Line 1", "scrap_percent"] == pytest.approx(5.0)    # 50 / 1000
    assert r.loc["Line 2", "scrap_percent"] == pytest.approx(8.0)    # 40 / 500


def rows_for(counts):
    out = []
    for (cause, shift), n in counts.items():
        out += [{"defect_type": cause, "shift": shift, "quantity": 1.0}] * n
    return pd.DataFrame(out)


def test_defect_mix_identical_proportions_gives_no_difference():
    # both shifts have the same 1 : 2 mix of A and B, so the test must see no difference
    d = rows_for({("A", "Day"): 10, ("B", "Day"): 20, ("A", "Night"): 20, ("B", "Night"): 40})
    m = defect_mix(d)
    assert m["p_value"] == pytest.approx(1.0)
    assert m["different"] is False
    assert m["table"].loc["A", "Night"] == 20


def test_defect_mix_opposite_proportions_gives_clear_difference():
    d = rows_for({("A", "Day"): 30, ("B", "Day"): 2, ("A", "Night"): 2, ("B", "Night"): 30})
    m = defect_mix(d)
    assert m["p_value"] < 0.001
    assert m["different"] is True
    assert m["cramers_v"] > 0.8


def test_lot_hours_and_scrap_per_100_hours():
    # lot X: 4 readings 30 minutes apart = 2 hours, 10 m scrapped = 500 m per 100 h
    # lot Y: 4 readings = 2 hours, 2 m scrapped = 100 m per 100 h
    times = pd.date_range("2026-01-01 06:00", periods=8, freq="30min").astype(str)
    log = pd.DataFrame({"timestamp": times, "line": "Line 1",
                        "lot": ["X"] * 4 + ["Y"] * 4})
    defects = pd.DataFrame({"line": ["Line 1"] * 3, "lot": ["X", "X", "Y"],
                            "defect_type": ["A", "A", "A"], "quantity": [6.0, 4.0, 2.0]})
    s = lot_summary(defects, log).set_index("lot")
    assert s.loc["X", "hours"] == 2
    assert s.loc["X", "scrap_m_per_100h"] == pytest.approx(500.0)
    assert s.loc["Y", "scrap_m_per_100h"] == pytest.approx(100.0)


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        pareto(pd.DataFrame({"defect_type": ["A"]}))      # no quantity column
    with pytest.raises(ValueError):
        pareto(make([]))                                   # no records
