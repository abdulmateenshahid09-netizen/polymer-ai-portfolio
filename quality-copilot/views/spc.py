"""SPC and capability: control chart, alarms, and Cp/Cpk for one characteristic."""
import numpy as np
import pandas as pd
import streamlit as st

import charts
import ui
from metrics.capability import capability
from metrics.spc import RULE_TEXT, alarm_events, spc_summary

st.title("SPC and capability")
filters = ui.get_filters()
meas = ui.table("measurements")
specs = ui.table("specs")
ui.need(meas, "No measurements yet. Add readings on the Data entry page, or load the demo data.")

pairs = meas[["product", "characteristic"]].drop_duplicates().merge(
    specs, on=["product", "characteristic"])
pairs = pairs[pairs["lsl"].notna() | pairs["usl"].notna()]
ui.need(pairs, "No specification limits are stored for the measured characteristics. "
               "Add them under Data entry > Specifications.")

c1, c2, c3, c4 = st.columns(4)
product = c1.selectbox("Product", sorted(pairs["product"].unique()))
chars = sorted(pairs.loc[pairs["product"] == product, "characteristic"])
characteristic = c2.selectbox("Characteristic", chars,
                              index=chars.index("wall_thickness") if "wall_thickness" in chars else 0)
method_label = c3.selectbox("Control limits method",
                            ["overall (suits drifting processes)", "moving_range (textbook)"])
method = method_label.split(" ")[0]
base_n = int(c4.number_input("Baseline readings", min_value=20, value=480, step=20,
                             help="The earliest readings of the line, used to set the control limits."))

spec = pairs[(pairs["product"] == product) & (pairs["characteristic"] == characteristic)].iloc[0]
lsl = None if pd.isna(spec["lsl"]) else float(spec["lsl"])
usl = None if pd.isna(spec["usl"]) else float(spec["usl"])
unit = spec["unit"] or ""

series = meas[(meas["product"] == product) & (meas["characteristic"] == characteristic)]
lines = [filters["line"]] if filters["line"] else sorted(series["line"].unique())
lines = [ln for ln in lines if ln in set(series["line"])]
ui.need(pd.DataFrame({"line": lines}), "No readings for this line, product and characteristic.")


def show_line(line):
    g = series[series["line"] == line].sort_values("timestamp").reset_index(drop=True)
    if len(g) < 3:
        st.info("At least 3 readings are needed for a control chart.")
        return
    x = g["value"].to_numpy()
    s = spc_summary(x, baseline=x[:min(base_n, len(x))], method=method)
    if filters["start"] is not None:
        mask = (g["timestamp"] >= filters["start"]).to_numpy()
    else:
        mask = np.ones(len(g), dtype=bool)
    if not mask.any():
        st.info("No readings in the chosen period.")
        return
    shown, rules = g[mask], s["rules"][mask]

    try:
        cap = capability(shown["value"], lsl, usl)
    except ValueError as error:
        cap = None
        st.warning(f"Capability cannot be calculated: {error}")

    if cap:
        def num(value):
            return "-" if value is None else f"{value:.2f}"
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Cpk (short-term)", num(cap["cpk"]), cap["cpk_rating"], delta_color="off", delta_arrow="off")
        k2.metric("Ppk (overall)", num(cap["ppk"]), cap["ppk_rating"], delta_color="off", delta_arrow="off")
        k3.metric("Cp", num(cap["cp"]))
        k4.metric("Pp", num(cap["pp"]))
        k5.metric("Out of spec", f"{cap['percent_below_lsl'] + cap['percent_above_usl']:.2f}%",
                  f"{cap['n']} readings", delta_color="off", delta_arrow="off")
        for text in cap["warnings"]:
            st.warning(text)
    for text in s["warnings"]:
        st.warning(text)

    fig = charts.control_chart(shown["timestamp"], shown["value"], rules, s["limits"],
                               lsl, usl, f"{line}: {characteristic}", unit)
    st.plotly_chart(fig)

    n_alarm = len(alarm_events(rules["alarm"]))
    n_warn = len(alarm_events(rules["warning"]))
    st.write(f"In this period: **{n_alarm} alarm events** (rules 1 and 2, act now) and "
             f"**{n_warn} drift warnings** (rules 3 and 4, watch closely).")

    hit = rules[rules["signal"]]
    if len(hit):
        table = pd.DataFrame({
            "time": shown.loc[hit.index, "timestamp"],
            "value": shown.loc[hit.index, "value"],
            "z-score": hit["z"],
            "level": np.where(hit["alarm"], "ALARM", "drift warning"),
            "rules": ["; ".join(RULE_TEXT[k] for k in RULE_TEXT if row[f"rule{k}"])
                      for _, row in hit.iterrows()],
        }).sort_values("time", ascending=False).head(15)
        st.subheader("Latest signals")
        st.dataframe(table, hide_index=True)

    lim = s["limits"]
    with st.expander("How the control limits were set"):
        st.write(f"Centre line {lim['center']:.3f}, sigma {lim['sigma']:.4f} ({lim['method']} method), "
                 f"from the earliest {lim['n']} readings of {line}. Control limits are the centre "
                 f"plus or minus 3 sigma. Neighbouring readings resemble each other "
                 f"(lag-1 correlation {s['lag1']:.2f}); above about 0.5 the 'overall' method is the "
                 "fairer choice.")
        st.write("Rules: " + "; ".join(f"{k} = {v}" for k, v in RULE_TEXT.items()))


if len(lines) > 1:
    for tab, line in zip(st.tabs(lines), lines):
        with tab:
            show_line(line)
else:
    show_line(lines[0])
