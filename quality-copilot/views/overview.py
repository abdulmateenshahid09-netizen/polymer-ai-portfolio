"""Overview: one status row per line."""
import numpy as np
import pandas as pd
import streamlit as st

import ui
from metrics.capability import capability
from metrics.oee import calculate_oee
from metrics.spc import alarm_events, spc_summary

st.title("Quality Analytics Copilot")
st.caption("Production quality at a glance. Use the filters on the left to change the line and period.")
filters = ui.get_filters()
meas, specs = ui.table("measurements"), ui.table("specs")
ui.need(meas, "No data yet. Load the demo data (python quality-copilot\\simulator.py) or add "
              "records on the Data entry page.")

pairs = meas[["product", "characteristic"]].drop_duplicates().merge(specs, on=["product", "characteristic"])
pairs = pairs[pairs["lsl"].notna() | pairs["usl"].notna()]
focus = None
if len(pairs):
    pick = pairs[pairs["characteristic"] == "wall_thickness"]
    focus = (pick if len(pick) else pairs).iloc[0]

prod_all = ui.apply(ui.table("production"), {**filters, "line": None}, "start_time")
defects_all = ui.apply(ui.table("defects"), {**filters, "line": None}, "timestamp")
shown_prod = ui.apply(ui.table("production"), filters, "start_time")
shown_def = ui.apply(ui.table("defects"), filters, "timestamp")

k1, k2, k3 = st.columns(3)
if len(shown_prod):
    result = calculate_oee(shown_prod)
    k1.metric("OEE", f"{result['oee'] * 100:.1f}%")
    k2.metric("Scrap rate", f"{shown_def['quantity'].sum() / shown_prod['total_count'].sum() * 100:.2f}%")
else:
    k1.metric("OEE", "-")
    k2.metric("Scrap rate", "-")
k3.metric("Period", filters["period"])

rows, attention = [], []
lines = [filters["line"]] if filters["line"] else ui.all_lines()
for line in lines:
    row = {"Line": line}
    if focus is not None:
        g = meas[(meas["line"] == line) & (meas["product"] == focus["product"]) &
                 (meas["characteristic"] == focus["characteristic"])].sort_values("timestamp")
        g = g.reset_index(drop=True)
        lsl = None if pd.isna(focus["lsl"]) else float(focus["lsl"])
        usl = None if pd.isna(focus["usl"]) else float(focus["usl"])
        mask = (g["timestamp"] >= filters["start"]).to_numpy() if filters["start"] is not None \
            else np.ones(len(g), dtype=bool)
        row["Readings"] = int(mask.sum())
        if mask.sum() >= 2:
            try:
                cap = capability(g.loc[mask, "value"], lsl, usl)
                row["Cpk"], row["Ppk"] = cap["cpk"], cap["ppk"]
                if cap["ppk"] is not None and cap["ppk"] < 1.0:
                    attention.append(f"{line}: Ppk {cap['ppk']:.2f}, overall capability is below 1.0.")
                elif cap["cpk"] is not None and cap["cpk"] < 1.33:
                    attention.append(f"{line}: Cpk {cap['cpk']:.2f}, below the usual 1.33 target.")
            except ValueError:
                pass
        if len(g) >= 20 and mask.any():
            x = g["value"].to_numpy()
            s = spc_summary(x, baseline=x[:min(480, len(x))], method="overall")
            row["Alarm events"] = len(alarm_events(s["rules"]["alarm"][mask]))
    p = ui.apply(ui.table("production"), {**filters, "line": line}, "start_time")
    d = ui.apply(ui.table("defects"), {**filters, "line": line}, "timestamp")
    if len(p):
        row["OEE %"] = round(calculate_oee(p)["oee"] * 100, 1)
        row["Scrap %"] = round(d["quantity"].sum() / p["total_count"].sum() * 100, 2)
    rows.append(row)

if focus is not None:
    st.subheader(f"Status by line ({focus['characteristic']})")
st.dataframe(pd.DataFrame(rows), hide_index=True)
if attention:
    st.subheader("Needs attention")
    for text in attention:
        st.warning(text)
elif rows:
    st.success("No line is below its capability targets in this period.")
st.caption("Cpk and Ppk use the readings in the chosen period. Alarm events are control chart "
           "alarms (rules 1 and 2) against limits set from the earliest 480 readings.")
