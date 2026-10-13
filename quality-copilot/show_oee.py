"""Print OEE by line, shift and day, plus where the planned time was lost."""
import pandas as pd

from db import read_table
from metrics.oee import calculate_oee, loss_breakdown, oee_by, BENCHMARK


def pct(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = (out[c] * 100).round(1)
    return out


def main():
    prod = read_table("production")
    down = read_table("downtime")
    cols = ["availability", "performance", "quality", "oee"]

    total = calculate_oee(prod)
    print(f"OEE over {total['records']} shift records "
          f"({total['planned_minutes']:.0f} planned minutes), all lines")
    print(f"  Availability {total['availability']*100:.1f}%   "
          f"Performance {total['performance']*100:.1f}%   "
          f"Quality {total['quality']*100:.1f}%   OEE {total['oee']*100:.1f}%")
    print(f"  Reference: availability {BENCHMARK['availability']*100:.0f}%, "
          f"performance {BENCHMARK['performance']*100:.0f}%, "
          f"quality {BENCHMARK['quality']*100:.0f}%, OEE {BENCHMARK['oee']*100:.0f}%\n")

    for by in ("line", "shift"):
        print(f"OEE by {by} (%)")
        print(pct(oee_by(prod, by), cols).to_string(index=False))
        print()

    daily = pct(oee_by(prod, "date"), cols)
    print("Best and worst day (%)")
    print(pd.concat([daily.nsmallest(2, "oee"), daily.nlargest(1, "oee")]).to_string(index=False))

    result = loss_breakdown(prod, down)
    print(f"\nWhere the planned time went (all lines, {result['planned_minutes']:.0f} minutes)")
    print(f"  Fully productive time: {result['productive_minutes']:.0f} minutes "
          f"({result['productive_percent']:.1f}% of planned)")
    print(result["losses"].to_string(index=False))


if __name__ == "__main__":
    main()
