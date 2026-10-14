"""AI insights: anomaly detection next to SPC alarms (more sections are added in later steps)."""
import numpy as np
import pandas as pd
import streamlit as st

import charts
import ui
from ai.anomaly import anomaly_events, detect_anomalies
from metrics.spc import alarm_events, spc_summary

BASELINE_READINGS = 480

st.title("AI insights")
st.caption("Demo data is simulated. The AI tools support the engineer; they do not replace "
           "the control chart or his judgement.")
filters = ui.get_filters()
meas = ui.table("measurements")
specs = ui.table("specs")
ui.need(meas, "No measurements yet. Add readings on the Data entry page, or load the demo data.")

wall = meas[meas["characteristic"] == "wall_thickness"]
ui.need(wall, "No wall thickness readings yet. The AI insights use wall thickness.")
products = sorted(wall["product"].unique())
product = st.selectbox("Product", products) if len(products) > 1 else products[0]
wall = wall[wall["product"] == product]
spec_row = specs[(specs["product"] == product) & (specs["characteristic"] == "wall_thickness")]
lsl = usl = None
if len(spec_row):
    lsl = None if pd.isna(spec_row.iloc[0]["lsl"]) else float(spec_row.iloc[0]["lsl"])
    usl = None if pd.isna(spec_row.iloc[0]["usl"]) else float(spec_row.iloc[0]["usl"])


@st.cache_data(show_spinner=False)
def analyse(values, baseline_n):
    """SPC rules and Isolation Forest on the same readings (cached: this is the heavy step)."""
    x = np.asarray(values, dtype=float)
    base = x[:min(baseline_n, len(x))]
    spc = spc_summary(x, baseline=base, method="overall")
    ai = detect_anomalies(x, baseline=base, contamination=0.01)
    return spc, ai


def event_table(shown, flags, score):
    """One row per group of consecutive flagged readings, newest first."""
    rows = []
    for start, end in anomaly_events(flags):
        part = slice(start, end + 1)
        rows.append({"start": shown["timestamp"].iloc[start],
                     "end": shown["timestamp"].iloc[end],
                     "readings": end - start + 1,
                     "highest value": float(shown["value"].iloc[part].max()),
                     "lowest value": float(shown["value"].iloc[part].min()),
                     "anomaly score": round(float(np.max(score[part])), 2)})
    return pd.DataFrame(rows).sort_values("start", ascending=False) if rows else pd.DataFrame()


def show_anomalies(line):
    g = wall[wall["line"] == line].sort_values("timestamp").reset_index(drop=True)
    if len(g) < BASELINE_READINGS // 4:
        st.info(f"At least {BASELINE_READINGS // 4} readings are needed for anomaly detection.")
        return
    try:
        spc, ai = analyse(tuple(g["value"]), BASELINE_READINGS)
    except ValueError as error:
        st.warning(f"Anomaly detection cannot run: {error}")
        return
    if filters["start"] is not None:
        mask = (g["timestamp"] >= filters["start"]).to_numpy()
    else:
        mask = np.ones(len(g), dtype=bool)
    if not mask.any():
        st.info("No readings in the chosen period.")
        return
    shown = g[mask].reset_index(drop=True)
    rules = spc["rules"][mask].reset_index(drop=True)
    flags = ai["anomaly"].to_numpy()[mask]
    score = ai["score"].to_numpy()[mask]

    n_spc = len(alarm_events(rules["alarm"]))
    n_ai = len(anomaly_events(flags))
    k1, k2, k3 = st.columns(3)
    k1.metric("SPC alarm events", n_spc, "rules 1 and 2", delta_color="off", delta_arrow="off")
    k2.metric("AI anomaly events", n_ai, "Isolation Forest", delta_color="off", delta_arrow="off")
    both = int((rules["alarm"].to_numpy() & flags).sum())
    k3.metric("Readings flagged by both", both, f"of {int(flags.sum())} AI flags",
              delta_color="off", delta_arrow="off")

    fig = charts.control_chart(shown["timestamp"], shown["value"], rules, spc["limits"],
                               lsl, usl, f"{line}: wall thickness", "mm")
    if flags.any():
        fig.add_scatter(x=shown["timestamp"][flags], y=shown["value"][flags], mode="markers",
                        name="AI anomaly", marker=dict(color="#9467bd", size=14, symbol="circle-open",
                                                       line=dict(width=2)),
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f} mm<br>AI anomaly<extra></extra>")
    st.plotly_chart(fig)

    events = event_table(shown, flags, score)
    if len(events):
        st.subheader("AI anomaly events")
        st.dataframe(events, hide_index=True)
    else:
        st.success("The AI found no unusual readings in this period.")


st.header("1. Anomalies: AI next to SPC rules")
lines = [filters["line"]] if filters["line"] else sorted(wall["line"].unique())
lines = [ln for ln in lines if ln in set(wall["line"])]
ui.need(pd.DataFrame({"line": lines}), "No wall thickness readings for this line.")
if len(lines) > 1:
    for tab, line in zip(st.tabs(lines), lines):
        with tab:
            show_anomalies(line)
else:
    show_anomalies(lines[0])

with st.expander("What this shows, honestly"):
    st.write("The model learns what normal looks like from the first 480 readings of each line, "
             "then flags readings that are unusual in level, spread or jump size. On the demo "
             "data it reacts as fast as the SPC rules 1 and 2 and raises modestly fewer false "
             "alarms (see `show_anomaly.py`). Use both: the chart rules tell you the rule "
             "that was broken, the AI flag also catches unusual spread.")
