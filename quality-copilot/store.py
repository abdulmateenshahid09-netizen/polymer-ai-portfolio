"""Save single entries typed into the app. Every function validates first and
returns (ok, message). Nothing is guessed: bad input is refused with a reason."""
import pandas as pd

from db import get_connection, init_db


def shift_of(ts):
    h = pd.Timestamp(ts).hour
    if 6 <= h < 14:
        return "Day"
    if 14 <= h < 22:
        return "Evening"
    return "Night"


def _text(value):
    value = (value or "").strip() if isinstance(value, str) or value is None else str(value).strip()
    return value or None


def _stamp(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _exists(conn, sql, params):
    return conn.execute(sql, params).fetchone() is not None


def get_spec(product, characteristic):
    init_db()
    conn = get_connection()
    row = conn.execute("SELECT lsl, usl, target, unit FROM specs WHERE product=? AND characteristic=?",
                       (product, characteristic)).fetchone()
    conn.close()
    return None if row is None else {"lsl": row[0], "usl": row[1], "target": row[2], "unit": row[3]}


def add_measurement(timestamp, line, product, characteristic, value, operator=None,
                    lot=None, notes=None, confirm_outlier=False):
    line, product, characteristic = _text(line), _text(product), _text(characteristic)
    if not (line and product and characteristic):
        return False, "Line, product and characteristic are required."
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False, "The value must be a number."
    if value != value:
        return False, "The value must be a number."

    spec = get_spec(product, characteristic)
    status = "no specification limits are stored for this characteristic"
    if spec and spec["lsl"] is not None and spec["usl"] is not None:
        width = spec["usl"] - spec["lsl"]
        if not (spec["lsl"] - width <= value <= spec["usl"] + width) and not confirm_outlier:
            return False, (f"{value} is far outside the specification ({spec['lsl']} to {spec['usl']}). "
                           "Check for a typing mistake, or tick the confirmation box if it is correct.")
        if value < spec["lsl"]:
            status = f"OUT OF SPEC: below the lower limit {spec['lsl']}"
        elif value > spec["usl"]:
            status = f"OUT OF SPEC: above the upper limit {spec['usl']}"
        else:
            status = f"within specification ({spec['lsl']} to {spec['usl']})"

    init_db()
    conn = get_connection()
    if _exists(conn, "SELECT 1 FROM measurements WHERE timestamp=? AND line=? AND characteristic=?",
               (_stamp(timestamp), line, characteristic)):
        conn.close()
        return False, "A reading for this line, characteristic and time already exists."
    conn.execute(
        "INSERT INTO measurements (timestamp, line, product, characteristic, value, unit, shift, "
        "operator, lot, notes) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (_stamp(timestamp), line, product, characteristic, value, spec["unit"] if spec else None,
         shift_of(timestamp), _text(operator), _text(lot), _text(notes)))
    conn.commit()
    conn.close()
    return True, f"Saved {characteristic} = {value}: {status}."


def add_production(start, end, line, product, planned_minutes, run_minutes,
                   ideal_rate_per_min, total_count, good_count):
    line, product = _text(line), _text(product)
    if not (line and product):
        return False, "Line and product are required."
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        return False, "The end time must be after the start time."
    if not planned_minutes > 0:
        return False, "Planned minutes must be above zero."
    if not 0 <= run_minutes <= planned_minutes:
        return False, "Run minutes must be between 0 and the planned minutes."
    if not ideal_rate_per_min > 0:
        return False, "The ideal rate must be above zero."
    if total_count < 0 or not 0 <= good_count <= total_count:
        return False, "Good count must be between 0 and the total count."

    init_db()
    conn = get_connection()
    if _exists(conn, "SELECT 1 FROM production WHERE start_time=? AND line=?", (_stamp(start), line)):
        conn.close()
        return False, "A production record for this line and start time already exists."
    conn.execute(
        "INSERT INTO production (start_time, end_time, line, product, planned_minutes, run_minutes, "
        "ideal_rate_per_min, total_count, good_count, shift) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (_stamp(start), _stamp(end), line, product, planned_minutes, run_minutes,
         ideal_rate_per_min, total_count, good_count, shift_of(start)))
    conn.commit()
    conn.close()
    availability = run_minutes / planned_minutes * 100
    performance = total_count / (ideal_rate_per_min * run_minutes) * 100 if run_minutes > 0 else 0
    quality = good_count / total_count * 100 if total_count > 0 else 0
    return True, (f"Saved. This run: availability {availability:.1f}%, performance "
                  f"{performance:.1f}%, quality {quality:.1f}%.")


def add_downtime(start, end, line, category, reason=None):
    line, category = _text(line), _text(category)
    if not (line and category):
        return False, "Line and category are required."
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        return False, "The end time must be after the start time."
    minutes = round((end - start).total_seconds() / 60, 1)
    init_db()
    conn = get_connection()
    if _exists(conn, "SELECT 1 FROM downtime WHERE start_time=? AND line=? AND category=?",
               (_stamp(start), line, category)):
        conn.close()
        return False, "This downtime event is already recorded."
    conn.execute(
        "INSERT INTO downtime (start_time, end_time, line, category, reason, minutes) "
        "VALUES (?,?,?,?,?,?)", (_stamp(start), _stamp(end), line, category, _text(reason), minutes))
    conn.commit()
    conn.close()
    return True, f"Saved {minutes:g} minutes of {category} downtime."


def add_defect(timestamp, line, product, defect_type, quantity, lot=None, notes=None):
    line, product, defect_type = _text(line), _text(product), _text(defect_type)
    if not (line and product and defect_type):
        return False, "Line, product and defect type are required."
    if not quantity > 0:
        return False, "The quantity must be above zero."
    init_db()
    conn = get_connection()
    if _exists(conn, "SELECT 1 FROM defects WHERE timestamp=? AND line=? AND defect_type=?",
               (_stamp(timestamp), line, defect_type)):
        conn.close()
        return False, "A defect of this type is already recorded at this time and line."
    conn.execute(
        "INSERT INTO defects (timestamp, line, product, defect_type, quantity, shift, lot, notes) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (_stamp(timestamp), line, product, defect_type, quantity, shift_of(timestamp),
         _text(lot), _text(notes)))
    conn.commit()
    conn.close()
    return True, f"Saved {quantity:g} m of {defect_type}."


def add_spec(product, characteristic, lsl, usl, target=None, unit=None):
    product, characteristic = _text(product), _text(characteristic)
    if not (product and characteristic):
        return False, "Product and characteristic are required."
    if lsl is None and usl is None:
        return False, "Give at least one limit."
    if lsl is not None and usl is not None and lsl >= usl:
        return False, "The lower limit must be smaller than the upper limit."
    init_db()
    conn = get_connection()
    conn.execute("INSERT OR REPLACE INTO specs (product, characteristic, lsl, usl, target, unit) "
                 "VALUES (?,?,?,?,?,?)", (product, characteristic, lsl, usl, target, _text(unit)))
    conn.commit()
    conn.close()
    return True, f"Saved limits for {product} / {characteristic}."
