"""Print capability for every line and shift, using the records in the database."""
import pandas as pd

from db import read_table
from metrics.capability import capability

CHARACTERISTIC = "wall_thickness"


def main():
    m = read_table("measurements")
    m = m[m["characteristic"] == CHARACTERISTIC].sort_values("timestamp")
    spec = read_table("specs")
    spec = spec[spec["characteristic"] == CHARACTERISTIC].iloc[0]
    print(f"{CHARACTERISTIC}: LSL {spec['lsl']}  USL {spec['usl']}  ({len(m)} readings)\n")

    rows = []
    for line, g in m.groupby("line"):
        groups = [("All shifts", g)] + [(s, gs) for s, gs in g.groupby("shift")]
        for name, data in groups:
            c = capability(data["value"], spec["lsl"], spec["usl"])
            rows.append({"line": line, "group": name, "n": c["n"], "mean": c["mean"],
                         "Cp": c["cp"], "Cpk": c["cpk"], "Pp": c["pp"], "Ppk": c["ppk"],
                         "Cpk rating": c["cpk_rating"], "% out": c["percent_below_lsl"] + c["percent_above_usl"]})
    print(pd.DataFrame(rows).to_string(index=False))

    c = capability(m[m["line"] == "Line 1"]["value"], spec["lsl"], spec["usl"])
    print("\nLine 1 notes:")
    for w in c["warnings"] or ["none"]:
        print(" -", w)


if __name__ == "__main__":
    main()
