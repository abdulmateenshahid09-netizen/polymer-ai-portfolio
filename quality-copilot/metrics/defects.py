"""Defect analysis: Pareto, scrap rates, defect mix by shift, and lot comparison."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def _need(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} data is missing columns: {', '.join(missing)}")
    if len(df) == 0:
        raise ValueError(f"No {name} records to analyse")


def pareto(defects, by="defect_type"):
    """Pareto table: biggest cause first, with cumulative share and the 'vital few'
    (the causes needed to reach 80% of the total scrap)."""
    _need(defects, [by, "quantity"], "defect")
    g = defects.groupby(by)["quantity"].agg(["sum", "count"])
    g.columns = ["quantity", "events"]
    g = g.sort_values("quantity", ascending=False)
    g["percent"] = g["quantity"] / g["quantity"].sum() * 100
    g["cumulative_percent"] = g["percent"].cumsum()
    g["vital_few"] = (g["cumulative_percent"] - g["percent"]) < 80
    g["quantity"] = g["quantity"].round(1)
    g[["percent", "cumulative_percent"]] = g[["percent", "cumulative_percent"]].round(1)
    return g.reset_index()


def scrap_rate(defects, production, by="line"):
    """Scrap as a percentage of everything produced, per line and/or shift."""
    keys = [by] if isinstance(by, str) else list(by)
    _need(defects, keys + ["quantity"], "defect")
    _need(production, keys + ["total_count"], "production")
    scrap = defects.groupby(keys)["quantity"].sum().rename("scrap")
    made = production.groupby(keys)["total_count"].sum().rename("produced")
    out = pd.concat([made, scrap], axis=1).fillna(0.0).reset_index()
    out["scrap_percent"] = (out["scrap"] / out["produced"] * 100).round(2)
    out["produced"] = out["produced"].round(0)
    out["scrap"] = out["scrap"].round(1)
    return out


def defect_mix(defects, rows="defect_type", cols="shift"):
    """Number of defect events for each cause (rows) in each group (cols), plus a
    chi-square test of whether the mix of causes really differs between groups."""
    _need(defects, [rows, cols], "defect")
    table = pd.crosstab(defects[rows], defects[cols])
    chi2, p_value, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0
    return {"table": table, "p_value": float(p_value), "cramers_v": round(cramers_v, 3),
            "different": bool(p_value < 0.05)}


def _lot_hours(process_log):
    """Running hours on each (line, lot): readings times the reading interval."""
    log = process_log.copy()
    log["timestamp"] = pd.to_datetime(log["timestamp"])
    step_hours = {}
    for line, g in log.groupby("line"):
        gaps = g["timestamp"].sort_values().diff().dropna()
        step_hours[line] = float(gaps.median().total_seconds() / 3600) if len(gaps) else 0.5
    log["hours"] = log["line"].map(step_hours)
    return log.groupby(["line", "lot"])["hours"].sum().rename("hours")


def lot_summary(defects, process_log, flag_z=3.0):
    """Scrap per 100 hours of running, for each resin lot, and which lots stand out.

    Hours on a lot come from the process log (readings times the reading interval).
    A lot is flagged when its scrap rate is far above the other lots (robust z-score).
    """
    _need(defects, ["line", "lot", "defect_type", "quantity"], "defect")
    _need(process_log, ["timestamp", "line", "lot"], "process log")
    hours = _lot_hours(process_log)

    scrap = defects.groupby(["line", "lot"])["quantity"].agg(["sum", "count"])
    scrap.columns = ["scrap_m", "events"]
    top = (defects.groupby(["line", "lot", "defect_type"])["quantity"].sum()
           .reset_index().sort_values("quantity", ascending=False)
           .drop_duplicates(["line", "lot"]).set_index(["line", "lot"])["defect_type"]
           .rename("main_defect"))

    out = pd.concat([hours, scrap, top], axis=1).fillna({"scrap_m": 0, "events": 0})
    out["scrap_m_per_100h"] = (out["scrap_m"] / out["hours"] * 100).round(1)
    med = out["scrap_m_per_100h"].median()
    mad = (out["scrap_m_per_100h"] - med).abs().median() * 1.4826
    out["robust_z"] = ((out["scrap_m_per_100h"] - med) / mad).round(1) if mad > 0 else 0.0
    out["flag"] = out["robust_z"] > flag_z
    out["hours"] = out["hours"].round(0)
    out["scrap_m"] = out["scrap_m"].round(1)
    return out.reset_index().sort_values("scrap_m_per_100h", ascending=False).reset_index(drop=True)


def lot_defect_outliers(defects, process_log, min_z=2.5):
    """Lots that are unusually bad for one specific defect type (robust z-score of
    scrap per 100 running hours, compared with the other lots)."""
    _need(defects, ["line", "lot", "defect_type", "quantity"], "defect")
    _need(process_log, ["timestamp", "line", "lot"], "process log")
    hours = _lot_hours(process_log)
    table = defects.pivot_table(index=["line", "lot"], columns="defect_type",
                                values="quantity", aggfunc="sum", fill_value=0)
    table = table.reindex(hours.index, fill_value=0)
    rate = table.div(hours, axis=0) * 100

    rows = []
    for defect_type in rate.columns:
        s = rate[defect_type]
        med = float(s.median())
        spread = float((s - med).abs().median() * 1.4826)
        if spread == 0:
            spread = float(s.std())
        if spread == 0:
            continue
        z = (s - med) / spread
        for (line, lot), value in s[z > min_z].items():
            rows.append({"line": line, "lot": lot, "defect_type": defect_type,
                         "scrap_m_per_100h": round(float(value), 1),
                         "typical_per_100h": round(med, 1), "robust_z": round(float(z[(line, lot)]), 1)})
    columns = ["line", "lot", "defect_type", "scrap_m_per_100h", "typical_per_100h", "robust_z"]
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("robust_z", ascending=False).reset_index(drop=True)
