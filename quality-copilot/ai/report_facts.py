"""Facts for the plain-language report: every number comes from the quality engine.

This module does NO writing and uses NO language model. It gathers the numbers for one
line and one period into a plain dictionary. The report writer may only use these numbers.
"""
import numpy as np
import pandas as pd

from ai.anomaly import anomaly_events, detect_anomalies
from metrics.capability import capability
from metrics.defects import pareto
from metrics.oee import calculate_oee, loss_breakdown
from metrics.spc import alarm_events, spc_summary

BASELINE_READINGS = 480     # same baseline as the dashboard pages
MIN_ANOMALY_READINGS = BASELINE_READINGS // 4
CHARACTERISTIC = "wall_thickness"
TOP_DEFECTS = 3


def _in_period(df, column, line, start):
    """Rows of one line from the start time on (start None = all data)."""
    if len(df) == 0:
        return df
    out = df[df["line"] == line]
    if start is not None:
        out = out[out[column] >= start]
    return out


def _spec(specs, product):
    row = specs[(specs["product"] == product) & (specs["characteristic"] == CHARACTERISTIC)]
    if len(row) == 0:
        return None, None, None
    r = row.iloc[0]

    def num(v):
        return None if pd.isna(v) else float(v)
    return num(r["lsl"]), num(r["usl"]), num(r["target"])


def _quality_section(g, specs, start, notes):
    """Capability, SPC alarms and AI anomalies for the wall thickness readings."""
    if len(g) == 0:
        notes.append("No wall thickness readings in this period.")
        return None
    product = g["product"].iloc[-1]
    g = g[g["product"] == product].sort_values("timestamp").reset_index(drop=True)
    lsl, usl, target = _spec(specs, product)
    mask = (np.ones(len(g), dtype=bool) if start is None
            else (g["timestamp"] >= start).to_numpy())
    shown = g[mask]
    out = {"product": product, "readings": int(len(shown)),
           "lsl_mm": lsl, "usl_mm": usl, "target_mm": target}
    if len(shown):
        out["mean_mm"] = round(float(shown["value"].mean()), 3)
    if lsl is None and usl is None:
        notes.append("No specification limits for wall thickness, so no capability numbers.")
    elif len(shown) >= 2:
        try:
            c = capability(shown["value"], lsl, usl)
            out.update({"cpk": c["cpk"], "ppk": c["ppk"], "cpk_rating": c["cpk_rating"],
                        "percent_out_of_spec": round(c["percent_below_lsl"]
                                                     + c["percent_above_usl"], 2)})
            notes.extend(c["warnings"])
        except ValueError as error:
            notes.append(f"Capability not calculated: {error}")
    if len(g) < MIN_ANOMALY_READINGS:
        notes.append(f"Fewer than {MIN_ANOMALY_READINGS} readings in total, so no alarm or "
                     "anomaly counts.")
    elif mask.any():
        x = g["value"].to_numpy()
        base = x[:min(BASELINE_READINGS, len(x))]
        try:
            spc = spc_summary(x, baseline=base, method="overall")
            flags = detect_anomalies(x, baseline=base, contamination=0.01)["anomaly"].to_numpy()
            out["spc_alarm_events"] = len(alarm_events(spc["rules"]["alarm"][mask]))
            out["ai_anomaly_events"] = len(anomaly_events(flags[mask]))
        except ValueError as error:
            notes.append(f"Alarm and anomaly counts not calculated: {error}")
    return out


def _production_section(prod, down, defects, notes):
    """OEE, biggest loss and scrap."""
    if len(prod) == 0:
        notes.append("No production records in this period.")
        return None
    r = calculate_oee(prod)
    out = {"oee_percent": round(r["oee"] * 100, 1),
           "availability_percent": round(r["availability"] * 100, 1),
           "performance_percent": round(r["performance"] * 100, 1),
           "quality_percent": round(r["quality"] * 100, 1),
           "planned_minutes": r["planned_minutes"]}
    notes.extend(r["warnings"])
    top = loss_breakdown(prod, down)["losses"].iloc[0]
    out["biggest_loss"] = str(top["loss"])
    out["biggest_loss_percent_of_planned"] = float(top["percent_of_planned"])
    if len(defects):
        out["scrap_percent"] = round(float(defects["quantity"].sum()
                                           / prod["total_count"].sum() * 100), 2)
        p = pareto(defects).head(TOP_DEFECTS)
        out["top_defects"] = [{"type": str(row["defect_type"]),
                               "percent_of_scrap": float(row["percent"])}
                              for _, row in p.iterrows()]
    else:
        out["scrap_percent"] = 0.0
        out["top_defects"] = []
    return out


def build_facts(line, period_label, start, end, meas, specs, production, downtime, defects):
    """All report numbers for one line. start None means all data. Tables are DataFrames
    with real timestamps (see ui.table)."""
    notes = []
    wall = meas[meas["characteristic"] == CHARACTERISTIC] if len(meas) else meas
    # all readings of the line: the baseline is the line's first readings, whatever the period
    quality = _quality_section(_in_period(wall, "timestamp", line, None), specs, start, notes)
    prod = _in_period(production, "start_time", line, start)
    down = _in_period(downtime, "start_time", line, start)
    dfct = _in_period(defects, "timestamp", line, start)
    return {
        "line": line, "period": period_label,
        "from": None if start is None else f"{start:%Y-%m-%d %H:%M}",
        "to": None if end is None else f"{end:%Y-%m-%d %H:%M}",
        "wall_thickness": quality,
        "production": _production_section(prod, down, dfct, notes),
        "notes": notes,
    }