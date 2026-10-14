"""Live demo: add simulated readings one at a time, as if a gauge were sending them.

The simulated readings are stored in the database like any other reading, with the
note 'simulated stream', so they can be removed again with delete_simulated()."""
import numpy as np
import pandas as pd

import store
from db import get_connection, init_db

NOTE = "simulated stream"


def next_value(history, rng, offset=0.0):
    """The next reading of a process that behaves like its own recent history.

    Uses a first-order autoregressive model fitted to the recent readings: the next
    value is pulled back towards the recent mean in proportion to the lag-1
    correlation, plus random noise sized to keep the same overall spread.
    `offset` shifts the process (used to inject a fault)."""
    x = np.asarray(history, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        raise ValueError("Need at least 10 earlier readings to simulate the next one")
    mean, std = float(x.mean()), float(x.std(ddof=1))
    r = float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], 0.0, 0.95)) if std > 0 else 0.0
    noise = std * float(np.sqrt(1 - r ** 2))
    return mean + offset + r * (float(x[-1]) - mean) + float(rng.normal(0, noise))


def append_simulated(line, product, characteristic, rng, offset=0.0, history_size=200):
    """Save one simulated reading, 30 minutes (or the usual step) after the last one."""
    init_db()
    conn = get_connection()
    rows = conn.execute(
        "SELECT timestamp, value, lot FROM measurements WHERE line=? AND product=? AND "
        "characteristic=? ORDER BY timestamp DESC LIMIT ?",
        (line, product, characteristic, history_size)).fetchall()
    conn.close()
    if len(rows) < 10:
        return False, "Need at least 10 earlier readings on this line to simulate more.", None, None
    stamps = pd.to_datetime([r[0] for r in rows[::-1]])
    values = [r[1] for r in rows[::-1]]
    step = pd.Series(stamps).diff().dropna().median()
    when = stamps[-1] + (step if pd.notna(step) and step > pd.Timedelta(0) else pd.Timedelta(minutes=30))
    value = round(next_value(values, rng, offset), 3)
    ok, message = store.add_measurement(when, line, product, characteristic, value,
                                        operator="Simulated", lot=rows[0][2], notes=NOTE,
                                        confirm_outlier=True)
    return ok, message, when, value


def delete_simulated():
    """Remove every reading the live demo created. Returns how many were removed."""
    init_db()
    conn = get_connection()
    cursor = conn.execute("DELETE FROM measurements WHERE notes = ?", (NOTE,))
    conn.commit()
    removed = cursor.rowcount
    conn.close()
    return removed


def count_simulated():
    init_db()
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM measurements WHERE notes = ?", (NOTE,)).fetchone()[0]
    conn.close()
    return n
