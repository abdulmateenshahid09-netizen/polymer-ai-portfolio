"""Settings recommender check: how good is the thickness prediction, and what does it advise?"""
import pandas as pd

from ai.recommender import (ALPHA, ewma_state, fit_and_validate, prepare, recommend_settings,
                            residuals)
from db import read_table

TARGET, LSL, USL = 10.4, 10.0, 11.1


def main():
    m = read_table("measurements")
    m = m[m["characteristic"] == "wall_thickness"]
    p = read_table("process_log")
    frames = {}
    for line, g in m.groupby("line"):
        frames[line] = prepare(g[["timestamp", "value"]],
                               p[p["line"] == line][["timestamp", "screw_rpm", "haul_speed_mpm",
                                                     "melt_temp_c", "mfr"]])
    model, r = fit_and_validate(frames)

    print("Process model (output per screw rpm, learned from the first 20 days):")
    print(f"  melt temperature: {model['melt_coef'] * 100:+.2f} % output per degree C")
    print(f"  resin MFR:        {model['mfr_coef'] * 10:+.1f} % output per 0.10 MFR unit\n")

    print("Predicting each new wall thickness reading, last 10 days "
          f"({r['n_test']} readings, {r['n_haul_changes']} right after a haul-off change):")
    print(pd.DataFrame([
        {"method": "Process model", "error, all readings (mm)": round(r["mae_model"], 3),
         "error, after a haul change (mm)": round(r["mae_model_haul_change"], 3)},
        {"method": "Next = last reading", "error, all readings (mm)": round(r["mae_persistence"], 3),
         "error, after a haul change (mm)": round(r["mae_persistence_haul_change"], 3)},
    ]).to_string(index=False))
    print(f"\nTypical prediction error (1 sigma): {r['sigma_mm']:.3f} mm\n")

    print("Example advice: the moment in the last 10 days when each line was furthest from target")
    for line, f in frames.items():
        f = f[pd.to_datetime(f["timestamp"]) >= pd.to_datetime(f["timestamp"]).min() + pd.Timedelta(days=20)]
        idx = (f["value"] - TARGET).abs().idxmax()
        full = frames[line]
        upto = full.loc[:idx]
        state = ewma_state(residuals(model, upto), ALPHA)
        row = full.loc[idx]
        rec = recommend_settings(model, state, row["melt_temp_c"], row["mfr"], row["screw_rpm"],
                                 row["haul_speed_mpm"], TARGET, LSL, USL, r["sigma_mm"])
        print(f"\n{line} at {row['timestamp']}: measured {row['value']:.2f} mm")
        print(f"  current haul-off {row['haul_speed_mpm']:.4f} m/min -> predicted thickness {rec['predicted_now']:.2f} mm")
        print(f"  advice: set haul-off to {rec['haul']:.4f} m/min ({rec['haul_change_pct']:+.1f} %) "
              f"-> predicted {rec['predicted_new']:.2f} mm")
        print(f"  distance to nearest spec limit: {rec['margin_sigmas']:.1f} sigma; "
              f"target reached: {'yes' if rec['reached_target'] else 'no'}; "
              f"at the change limit: {'yes' if rec['at_limit'] else 'no'}")
    print("\nDemo data only. A real line needs supervised trials before any advice is used.")


if __name__ == "__main__":
    main()
