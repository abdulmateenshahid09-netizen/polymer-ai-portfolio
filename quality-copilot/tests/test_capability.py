"""Capability tests. Every expected number below was calculated by hand.

Readings: 10.0, 10.2, 10.0, 10.2, 10.0, 10.2
  mean = 10.1
  every moving range = 0.2, so average moving range = 0.2
  sigma within = 0.2 / 1.128 = 0.17730
  overall sample std = sqrt(0.06 / 5) = 0.10954
"""
import pytest

from metrics.capability import capability, rating

X = [10.0, 10.2, 10.0, 10.2, 10.0, 10.2]


def test_centred_process():
    # limits 9.5 to 10.7: Cp = 1.2*1.128/(6*0.2) = 1.128, centred so Cpk is the same
    c = capability(X, lsl=9.5, usl=10.7)
    assert c["mean"] == pytest.approx(10.1)
    assert c["sigma_within"] == pytest.approx(0.2 / 1.128, abs=1e-4)
    assert c["sigma_overall"] == pytest.approx(0.109545, abs=1e-4)
    assert c["cp"] == pytest.approx(1.128, abs=1e-3)
    assert c["cpk"] == pytest.approx(1.128, abs=1e-3)
    # Pp = 1.2 / (6 * 0.109545) = 1.8257
    assert c["pp"] == pytest.approx(1.826, abs=1e-3)
    assert c["ppk"] == pytest.approx(1.826, abs=1e-3)


def test_off_centre_process():
    # limits 9.8 to 10.7: Cp = 0.9*1.128/1.2 = 0.846
    # Cpu = 0.6*1.128/0.6 = 1.128 ; Cpl = 0.3*1.128/0.6 = 0.564 ; Cpk = the smaller one
    c = capability(X, lsl=9.8, usl=10.7)
    assert c["cp"] == pytest.approx(0.846, abs=1e-3)
    assert c["cpk"] == pytest.approx(0.564, abs=1e-3)
    assert c["cpk"] < c["cp"]


def test_one_sided_limit():
    c = capability(X, usl=10.7)
    assert c["cp"] is None and c["pp"] is None
    assert c["cpk"] == pytest.approx(1.128, abs=1e-3)


def test_out_of_spec_percent():
    # limits 10.05 to 10.15: the three 10.0 readings are below, the three 10.2 are above
    c = capability(X, lsl=10.05, usl=10.15)
    assert c["percent_below_lsl"] == pytest.approx(50.0)
    assert c["percent_above_usl"] == pytest.approx(50.0)


def test_small_sample_warning():
    c = capability(X, lsl=9.5, usl=10.7)
    assert any("fewer than 30" in w for w in c["warnings"])


def test_drift_warning_when_ppk_much_lower_than_cpk():
    # slow drift: neighbouring readings are close (small moving range) but the
    # series wanders widely, so Ppk is far below Cpk
    drift = [10.0 + 0.02 * i for i in range(40)]
    c = capability(drift, lsl=9.0, usl=11.5)
    assert c["ppk"] < 0.8 * c["cpk"]
    assert any("drifts" in w for w in c["warnings"])


def test_missing_values_are_ignored():
    with_gap = X + [float("nan")]
    assert capability(with_gap, 9.5, 10.7)["n"] == 6


@pytest.mark.parametrize("value, expected", [
    (1.67, "excellent"), (1.40, "capable"), (1.33, "capable"),
    (1.10, "marginal"), (1.00, "marginal"), (0.99, "not capable"), (None, "not available"),
])
def test_rating(value, expected):
    assert rating(value) == expected


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        capability(X)                              # no limits
    with pytest.raises(ValueError):
        capability(X, lsl=10.7, usl=9.5)           # limits the wrong way round
    with pytest.raises(ValueError):
        capability([10.0], lsl=9.5, usl=10.7)      # one reading only
    with pytest.raises(ValueError):
        capability([10.0] * 5, lsl=9.5, usl=10.7)  # no variation at all
