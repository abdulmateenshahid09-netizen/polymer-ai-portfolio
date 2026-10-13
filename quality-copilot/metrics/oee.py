"""OEE (overall equipment effectiveness) and loss breakdown. Python does the maths."""
import pandas as pd

# commonly quoted "world class" levels, used only as a reference line
BENCHMARK = {"availability": 0.90, "performance": 0.95, "quality": 0.99, "oee": 0.85}

REQUIRED = ["planned_minutes", "run_minutes", "ideal_rate_per_min", "total_count", "good_count"]


def _check(production):
    missing = [c for c in REQUIRED if c not in production.columns]
    if missing:
        raise ValueError(f"Production data is missing columns: {', '.join(missing)}")
    if len(production) == 0:
        raise ValueError("No production records to calculate OEE from")


def calculate_oee(production):
    """OEE from production records. Totals are summed first, then divided, so long
    shifts count more than short ones (never average the ratios of each shift)."""
    _check(production)
    p = production
    planned = float(p["planned_minutes"].sum())
    run = float(p["run_minutes"].sum())
    ideal_output = float((p["ideal_rate_per_min"] * p["run_minutes"]).sum())
    total = float(p["total_count"].sum())
    good = float(p["good_count"].sum())
    if planned <= 0 or run <= 0 or ideal_output <= 0 or total <= 0:
        raise ValueError("Planned time, run time and output must all be above zero")

    availability = run / planned
    performance = total / ideal_output
    quality = good / total
    oee = availability * performance * quality

    warnings = []
    if performance > 1:
        warnings.append("Performance is above 100%: the ideal rate is set too low.")
    if good > total:
        warnings.append("Good count is larger than total count: check the records.")
    return {
        "records": int(len(p)),
        "planned_minutes": round(planned, 1), "run_minutes": round(run, 1),
        "total_count": round(total, 1), "good_count": round(good, 1),
        "availability": round(availability, 4), "performance": round(performance, 4),
        "quality": round(quality, 4), "oee": round(oee, 4),
        "warnings": warnings,
    }


def loss_breakdown(production, downtime=None):
    """Where the planned time went: downtime by category, speed loss and scrap.

    All losses are in minutes of planned time, so they add up with the fully
    productive time to exactly the planned time.
    """
    _check(production)
    p = production
    planned = float(p["planned_minutes"].sum())
    downtime_total = float((p["planned_minutes"] - p["run_minutes"]).sum())

    rows, classified = [], 0.0
    if downtime is not None and len(downtime):
        by_category = downtime.groupby("category")["minutes"].sum()
        for category, minutes in by_category.items():
            rows.append(("Availability", category, float(minutes)))
        classified = float(by_category.sum())
    if downtime_total - classified > 0.5:
        rows.append(("Availability", "Unclassified downtime", downtime_total - classified))

    speed = float((p["run_minutes"] - p["total_count"] / p["ideal_rate_per_min"]).sum())
    scrap = float(((p["total_count"] - p["good_count"]) / p["ideal_rate_per_min"]).sum())
    rows.append(("Performance", "Speed loss (below ideal rate)", speed))
    rows.append(("Quality", "Scrap and rework", scrap))

    losses = pd.DataFrame(rows, columns=["type", "loss", "minutes"])
    losses["minutes"] = losses["minutes"].round(1)
    losses["percent_of_planned"] = (losses["minutes"] / planned * 100).round(2)
    losses = losses.sort_values("minutes", ascending=False).reset_index(drop=True)

    productive = float((p["good_count"] / p["ideal_rate_per_min"]).sum())
    return {"losses": losses, "planned_minutes": round(planned, 1),
            "productive_minutes": round(productive, 1),
            "productive_percent": round(productive / planned * 100, 2)}


def oee_by(production, by="line"):
    """OEE for each line, shift or date. Returns one row per group."""
    _check(production)
    p = production.copy()
    if by == "date":
        p["date"] = p["start_time"].astype(str).str[:10]
    if by not in p.columns:
        raise ValueError(f"Cannot group by '{by}'")
    rows = []
    for key, g in p.groupby(by):
        r = calculate_oee(g)
        rows.append({by: key, "records": r["records"], "availability": r["availability"],
                     "performance": r["performance"], "quality": r["quality"], "oee": r["oee"]})
    return pd.DataFrame(rows)
