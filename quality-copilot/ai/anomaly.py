"""Anomaly detection for readings in time order (Isolation Forest on rolling features).

The model learns what NORMAL looks like from a baseline of readings, then flags
readings that look unusual in level, spread or jump size. It complements the SPC rules.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from metrics.spc import alarm_events

FEATURES = ["z", "roll_mean", "roll_std", "diff"]
WINDOW = 8          # rolling window, in readings (4 hours at one reading per 30 minutes)
MIN_BASELINE = 30


def build_features(values, baseline):
    """Four numbers per reading, all in units of baseline standard deviations.

    z: distance from the baseline mean. roll_mean: average z over the last readings.
    roll_std: spread of z over the last readings. diff: change from the previous reading.
    """
    x = pd.Series(np.asarray(values, dtype=float))
    base = np.asarray(baseline, dtype=float)
    base = base[~np.isnan(base)]
    if len(base) < MIN_BASELINE:
        raise ValueError(f"Need at least {MIN_BASELINE} baseline readings")
    sd = base.std(ddof=1)
    if sd == 0:
        raise ValueError("Baseline readings are all identical")
    z = (x - base.mean()) / sd
    out = pd.DataFrame({
        "z": z,
        "roll_mean": z.rolling(WINDOW, min_periods=3).mean(),
        "roll_std": z.rolling(WINDOW, min_periods=3).std(),
        "diff": z.diff(),
    })
    return out.bfill().fillna(0.0)


def detect_anomalies(values, baseline=None, contamination=0.01, seed=0):
    """Flag unusual readings. baseline: readings that define normal (default: first 480).

    contamination: share of baseline readings the model may call unusual (0.01 = 1 %).
    Lower means fewer false alarms but a slower reaction.
    """
    x = np.asarray(values, dtype=float)
    if baseline is None:
        baseline = x[:480]
    feats = build_features(x, baseline)
    n_base = min(len(baseline), len(x))
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    model.fit(feats.iloc[:n_base][FEATURES])
    out = feats.copy()
    out["score"] = -model.score_samples(feats[FEATURES])   # higher = more unusual
    out["anomaly"] = model.predict(feats[FEATURES]) == -1
    return out


def anomaly_events(flags):
    """Group consecutive flagged readings into (start, end) positions."""
    return alarm_events(flags)


def evaluate_detection(flags, timestamps, faults, grace_hours=2):
    """Compare flags with known fault windows.

    faults: list of (start, end) timestamps. Readings within grace_hours after a fault
    ends are not counted as false alarms (the process is still recovering).
    Returns detection delay in minutes per fault (None = missed) and false alarm events per day.
    """
    flags = np.asarray(flags, dtype=bool)
    ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    in_window = np.zeros(len(ts), dtype=bool)
    delays = []
    for start, end in faults:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        in_window |= ((ts >= start) & (ts < end + pd.Timedelta(hours=grace_hours))).to_numpy()
        hits = ts[flags & ((ts >= start) & (ts < end)).to_numpy()]
        delays.append(int((hits.iloc[0] - start).total_seconds() // 60) if len(hits) else None)
    outside = ~in_window
    false_events = len(alarm_events(flags & outside))
    days = outside.sum() * _step_days(ts)
    return {
        "delays_min": delays,
        "detected": sum(d is not None for d in delays),
        "faults": len(delays),
        "false_alarm_events": false_events,
        "false_alarms_per_day": round(false_events / days, 2) if days > 0 else float("nan"),
    }


def _step_days(ts):
    if len(ts) < 2:
        return 0.0
    step = ts.diff().dropna().median()
    return step.total_seconds() / 86400.0
