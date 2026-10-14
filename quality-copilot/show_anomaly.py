"""Anomaly detection check: does it catch the simulated faults, and how many false alarms?"""
import pandas as pd

from ai.anomaly import detect_anomalies, evaluate_detection
from db import read_table, DB_PATH
from metrics.spc import spc_summary

CHARACTERISTIC = "wall_thickness"
BASELINE_READINGS = 480


def main():
    m = read_table("measurements")
    m = m[m["characteristic"] == CHARACTERISTIC].sort_values("timestamp")
    faults = pd.read_csv(DB_PATH.parent / "sim_events.csv", parse_dates=["start", "end"])
    faults = faults[faults["end"].notna()]

    rows = []
    for line, g in m.groupby("line"):
        g = g.reset_index(drop=True)
        x = g["value"].to_numpy()
        ts = pd.to_datetime(g["timestamp"])
        windows = [(f["start"], f["end"]) for _, f in faults[faults["line"] == line].iterrows()]

        spc = spc_summary(x, baseline=x[:BASELINE_READINGS], method="overall")
        ai = detect_anomalies(x, baseline=x[:BASELINE_READINGS], contamination=0.01)
        for name, flags in [("SPC rules 1-2", spc["rules"]["alarm"]),
                            ("Isolation Forest", ai["anomaly"])]:
            r = evaluate_detection(flags, ts, windows)
            rows.append({"line": line, "method": name,
                         "faults caught": f"{r['detected']} of {r['faults']}",
                         "delays (min)": str(r["delays_min"]),
                         "false alarms per day": r["false_alarms_per_day"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nFalse alarms = flagged groups outside the fault windows (2 hours of recovery allowed).")


if __name__ == "__main__":
    main()
