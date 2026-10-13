"""Control chart tests. Rule tests use limits with centre 0 and sigma 1, so each
reading is its own z-score and the expected flags can be checked by eye."""
import pytest

from metrics.spc import (control_limits, check_rules, alarm_events,
                         lag1_autocorrelation, spc_summary)

UNIT = {"center": 0.0, "sigma": 1.0}
X = [10.0, 10.2, 10.0, 10.2, 10.0, 10.2]


def flags(values, rule):
    return list(check_rules(values, UNIT)[rule])


def test_limits_moving_range_method():
    # sigma = 0.2 / 1.128 = 0.177305 ; UCL = 10.1 + 3*0.177305 = 10.6319
    lim = control_limits(X, "moving_range")
    assert lim["center"] == pytest.approx(10.1)
    assert lim["sigma"] == pytest.approx(0.177305, abs=1e-5)
    assert lim["ucl"] == pytest.approx(10.6319, abs=1e-3)
    assert lim["lcl"] == pytest.approx(9.5681, abs=1e-3)
    # moving range chart: centre 0.2, upper limit 3.267 * 0.2 = 0.6534
    assert lim["mr_ucl"] == pytest.approx(0.6534, abs=1e-4)


def test_limits_overall_method():
    # sigma = 0.109545 ; UCL = 10.1 + 0.328634 = 10.4286
    lim = control_limits(X, "overall")
    assert lim["sigma"] == pytest.approx(0.109545, abs=1e-5)
    assert lim["ucl"] == pytest.approx(10.4286, abs=1e-3)


def test_rule1_point_beyond_3_sigma():
    assert flags([0, 0, 3.5, 0], "rule1") == [False, False, True, False]
    assert flags([0, -3.5, 0], "rule1") == [False, True, False]
    assert not any(flags([0, 2.9, -2.9], "rule1"))


def test_rule2_two_of_three_beyond_2_sigma():
    # readings 1 and 3 are above 2 sigma: flagged when the second one arrives
    assert flags([0, 2.5, 0, 2.5, 0], "rule2") == [False, False, False, True, False]
    # one point beyond 2 sigma alone is not enough
    assert not any(flags([0, 2.5, 0, 0, 0], "rule2"))
    # opposite sides do not count together
    assert not any(flags([2.5, 0, -2.5], "rule2"))


def test_rule3_four_of_five_beyond_1_sigma():
    assert flags([1.5, 1.5, 0, 1.5, 1.5], "rule3")[4] is True
    assert not any(flags([1.5, 1.5, 0, 0, 1.5], "rule3"))


def test_rule4_eight_in_a_row_one_side():
    assert flags([0.5] * 8, "rule4")[7] is True
    assert not any(flags([0.5] * 7, "rule4"))
    assert not any(flags([0.5] * 7 + [-0.5], "rule4"))
    assert flags([-0.5] * 9, "rule4")[7:] == [True, True]


def test_alarm_versus_warning():
    out = check_rules([0, 3.5, 0], UNIT)
    assert list(out["alarm"]) == [False, True, False]
    drift = check_rules([0.5] * 8, UNIT)
    assert drift["warning"].iloc[7] and not drift["alarm"].iloc[7]


def test_alarm_events_group_consecutive_points():
    assert alarm_events([False, True, True, False, True]) == [(1, 2), (4, 4)]
    assert alarm_events([True, True]) == [(0, 1)]
    assert alarm_events([False, False]) == []


def test_lag1_autocorrelation():
    assert lag1_autocorrelation([1, 2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert lag1_autocorrelation([1, -1, 1, -1, 1, -1]) == pytest.approx(-1.0)


def test_summary_counts_alarm_events():
    base = [10.0, 10.2] * 10                    # 20 readings for the limits
    values = base + [11.0] + base               # one wild reading in the middle
    s = spc_summary(values, baseline=base, method="overall")
    assert s["alarm_events"] == 1
    assert s["points_beyond_limits"] == 1


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        control_limits([10.0, 10.1])                      # too few readings
    with pytest.raises(ValueError):
        control_limits(X, "magic")                        # unknown method
    with pytest.raises(ValueError):
        control_limits([10.0] * 5)                        # no variation
