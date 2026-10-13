"""Control chart limits and Western Electric rules for individual readings."""
import numpy as np
import pandas as pd

D2 = 1.128        # constant for moving ranges of 2 readings
D4 = 3.267        # upper limit factor for the moving range chart

RULE_TEXT = {
    1: "1 point beyond 3 sigma",
    2: "2 of 3 points beyond 2 sigma, same side",
    3: "4 of 5 points beyond 1 sigma, same side",
    4: "8 points in a row on one side of the centre line",
}


def _clean(values):
    x = np.asarray(values, dtype=float)
    return x[~np.isnan(x)]


def lag1_autocorrelation(values):
    """How much each reading resembles the one before it (0 = independent)."""
    x = _clean(values)
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def control_limits(values, method="moving_range"):
    """Centre line and 3-sigma limits from a baseline of readings in time order.

    method 'moving_range': sigma = average moving range / 1.128 (textbook I-MR chart).
    method 'overall': sigma = standard deviation of the baseline. Use this when the
    readings drift slowly (autocorrelated), because moving-range limits are then too tight.
    """
    if method not in ("moving_range", "overall"):
        raise ValueError("method must be 'moving_range' or 'overall'")
    x = _clean(values)
    if len(x) < 3:
        raise ValueError("Need at least 3 readings to set control limits")
    center = float(x.mean())
    mr_bar = float(np.abs(np.diff(x)).mean())
    sigma = mr_bar / D2 if method == "moving_range" else float(x.std(ddof=1))
    if sigma == 0:
        raise ValueError("All readings are identical, so limits cannot be set")
    return {
        "method": method, "n": int(len(x)), "center": center, "sigma": sigma,
        "ucl": center + 3 * sigma, "lcl": center - 3 * sigma,
        "mr_center": mr_bar, "mr_ucl": D4 * mr_bar, "mr_lcl": 0.0,
    }


def check_rules(values, limits):
    """Apply the four Western Electric rules. Returns one row per reading."""
    x = np.asarray(values, dtype=float)
    z = pd.Series((x - limits["center"]) / limits["sigma"])

    above2, below2 = (z > 2).astype(int), (z < -2).astype(int)
    above1, below1 = (z > 1).astype(int), (z < -1).astype(int)
    above0, below0 = (z > 0).astype(int), (z < 0).astype(int)

    out = pd.DataFrame({"value": x, "z": z.round(2)})
    out["rule1"] = z.abs() > 3
    out["rule2"] = (((above2.rolling(3).sum() >= 2) & (z > 2)) |
                    ((below2.rolling(3).sum() >= 2) & (z < -2)))
    out["rule3"] = (((above1.rolling(5).sum() >= 4) & (z > 1)) |
                    ((below1.rolling(5).sum() >= 4) & (z < -1)))
    out["rule4"] = (above0.rolling(8).sum() == 8) | (below0.rolling(8).sum() == 8)
    # rules 1 and 2 are action alarms; rules 3 and 4 are early drift warnings
    out["alarm"] = out["rule1"] | out["rule2"]
    out["warning"] = (out["rule3"] | out["rule4"]) & ~out["alarm"]
    out["signal"] = out["alarm"] | out["warning"]
    return out


def alarm_events(flags):
    """Group consecutive flagged readings into events: list of (start, end) positions."""
    flags = np.asarray(flags, dtype=bool)
    events, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(flags) - 1))
    return events


def spc_summary(values, baseline=None, method="moving_range"):
    """Limits, rule violations and warnings for a series of readings.

    baseline: readings used to set the limits (default: all readings).
    """
    x = _clean(values)
    limits = control_limits(x if baseline is None else baseline, method)
    rules = check_rules(x, limits)
    events = alarm_events(rules["alarm"])
    warning_events = alarm_events(rules["warning"])

    lag1 = lag1_autocorrelation(x)
    warnings = []
    if limits["n"] < 20:
        warnings.append(f"Limits come from only {limits['n']} readings; "
                        "use at least 20 for reliable limits.")
    if lag1 > 0.5 and method == "moving_range":
        warnings.append(f"Readings are strongly autocorrelated (lag-1 r = {lag1:.2f}), so "
                        "moving-range limits are too tight and give many false alarms. "
                        "Use method='overall'.")
    return {
        "limits": limits, "n": int(len(x)), "lag1": round(lag1, 2),
        "points_beyond_limits": int(rules["rule1"].sum()),
        "alarm_points": int(rules["alarm"].sum()),
        "warning_points": int(rules["warning"].sum()),
        "alarm_events": len(events), "events": events,
        "warning_events": len(warning_events),
        "rule_counts": {k: int(rules[f"rule{k}"].sum()) for k in RULE_TEXT},
        "rules": rules, "warnings": warnings,
    }
