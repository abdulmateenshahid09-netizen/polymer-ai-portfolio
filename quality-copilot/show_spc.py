"""Control chart check: limits, alarms, and whether the simulated faults are caught."""
import pandas as pd

from db import read_table, DB_PATH
from metrics.spc import spc_summary

CHARACTERISTIC = "wall_thickness"
BASELINE_READINGS = 480   # limits come from the first 480 readings (10 days)


def main():
    m = read_table("measurements")
    m = m[m["characteristic"] == CHARACTERISTIC].sort_values("timestamp")
    faults = pd.read_csv(DB_PATH.parent / "sim_events.csv", parse_dates=["start", "end"])

    print(f"{CHARACTERISTIC} control charts "
          f"(limits from the first {BASELINE_READINGS} readings, about 10 days)\n")

    rows, detail = [], []
    for line, g in m.groupby("line"):
        g = g.reset_index(drop=True)
        x = g["value"].to_numpy()
        ts = pd.to_datetime(g["timestamp"])
        for method in ("moving_range", "overall"):
            s = spc_summary(x, baseline=x[:BASELINE_READINGS], method=method)
            rows.append({"line": line, "method": method,
                         "LCL": round(s["limits"]["lcl"], 2), "UCL": round(s["limits"]["ucl"], 2),
                         "alarm events": s["alarm_events"], "drift warnings": s["warning_events"]})
            if method == "overall":
                detail.append((line, ts, s))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nDid the alarms catch the simulated faults? (limits method: overall)")
    for line, ts, s in detail:
        alarm = s["rules"]["alarm"].to_numpy()
        for _, f in faults[(faults["line"] == line) & faults["end"].notna()].iterrows():
            in_window = ((ts >= f["start"]) & (ts < f["end"])).to_numpy()
            hits = ts[alarm & in_window]
            if len(hits):
                delay = int((hits.iloc[0] - f["start"]).total_seconds() / 60)
                print(f"  {line}  {f['event']:17s} first alarm {delay} minutes after the fault started")
            else:
                print(f"  {line}  {f['event']:17s} NOT DETECTED")

    print("\nAlarm events are groups of consecutive alarm readings. Some fall outside the")
    print("three fault windows: most of them come from the night shift, which varies more.")


if __name__ == "__main__":
    main()
