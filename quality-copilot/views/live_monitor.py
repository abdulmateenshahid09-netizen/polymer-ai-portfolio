"""Live monitor: a control chart that refreshes by itself, with an optional demo stream."""
import numpy as np
import pandas as pd
import streamlit as st

import charts
import live
import ui
from metrics.spc import spc_summary

st.title("Live monitor")
st.caption("Turn on live mode and the chart refreshes by itself. Turn on the demo stream and "
           "simulated readings arrive one at a time, as if a gauge were sending them.")
filters = ui.get_filters()
meas, specs = ui.table("measurements"), ui.table("specs")
ui.need(meas, "No measurements yet. Load the demo data or add readings on the Data entry page.")

pairs = meas[["product", "characteristic"]].drop_duplicates().merge(specs, on=["product", "characteristic"])
pairs = pairs[pairs["lsl"].notna() & pairs["usl"].notna()]
ui.need(pairs, "Live monitoring needs a characteristic with both specification limits.")

c1, c2, c3 = st.columns(3)
product = c1.selectbox("Product", sorted(pairs["product"].unique()))
chars = sorted(pairs.loc[pairs["product"] == product, "characteristic"])
characteristic = c2.selectbox("Characteristic", chars,
                              index=chars.index("wall_thickness") if "wall_thickness" in chars else 0)
shown_points = int(c3.number_input("Readings shown", min_value=30, max_value=500, value=120, step=10))
spec = pairs[(pairs["product"] == product) & (pairs["characteristic"] == characteristic)].iloc[0]
lsl, usl, unit = float(spec["lsl"]), float(spec["usl"]), spec["unit"] or ""

t1, t2, t3 = st.columns(3)
live_on = t1.toggle("Live mode (refresh automatically)", key="live_on")
stream_on = t2.toggle("Demo stream (adds simulated readings)", key="live_stream",
                      help="Writes simulated readings into the database. Remove them afterwards "
                           "with the button below.")
seconds = t3.select_slider("Refresh every (seconds)", [2, 3, 5, 10], value=3)

def remove_simulated():
    """Delete the simulated readings and switch live mode and the stream off."""
    st.session_state["removed_count"] = live.delete_simulated()
    st.session_state["live_stream"] = False
    st.session_state["live_on"] = False
    st.session_state["fault_left"] = 0


b1, b2, b3 = st.columns(3)
if b1.button("Inject a fault into the demo stream"):
    st.session_state["fault_left"] = 6
b2.button("Remove simulated readings", on_click=remove_simulated)
removed = st.session_state.pop("removed_count", None)
if removed is not None:
    st.success(f"Removed {removed} simulated readings and switched the stream off.")
b3.caption(f"Simulated readings in the database: {live.count_simulated()}")

series_all = meas[(meas["product"] == product) & (meas["characteristic"] == characteristic)]
lines = [filters["line"]] if filters["line"] else sorted(series_all["line"].unique())
rng = np.random.default_rng()


@st.fragment(run_every=seconds if live_on else None)
def monitor():
    data = ui.table("measurements")
    series = data[(data["product"] == product) & (data["characteristic"] == characteristic)]
    fault_size = 0.35 * (usl - lsl)

    if live_on and stream_on:
        for line in lines:
            offset = fault_size if st.session_state.get("fault_left", 0) > 0 else 0.0
            live.append_simulated(line, product, characteristic, rng, offset)
        if st.session_state.get("fault_left", 0) > 0:
            st.session_state["fault_left"] -= 1
        series = ui.table("measurements")
        series = series[(series["product"] == product) & (series["characteristic"] == characteristic)]

    st.write(f"{'LIVE' if live_on else 'Paused'}"
             + (" - demo stream on" if live_on and stream_on else "")
             + (f" - fault active for {st.session_state['fault_left']} more readings"
                if st.session_state.get("fault_left", 0) > 0 and stream_on else ""))
    columns = st.columns(len(lines)) if len(lines) > 1 else [st.container()]
    for column, line in zip(columns, lines):
        g = series[series["line"] == line].sort_values("timestamp").reset_index(drop=True)
        with column:
            if len(g) < 20:
                st.info(f"{line}: at least 20 readings are needed.")
                continue
            x = g["value"].to_numpy()
            s = spc_summary(x, baseline=x[:min(480, len(x))], method="overall")
            tail = slice(max(len(g) - shown_points, 0), None)
            last, rules = float(x[-1]), s["rules"].iloc[-1]
            status = ("OUT OF SPEC" if (last < lsl or last > usl) else "within specification")
            m1, m2 = st.columns(2)
            m1.metric(f"{line}: latest reading", f"{last:.3f} {unit}", status,
                      delta_color="off", delta_arrow="off")
            m2.metric("Time", f"{g['timestamp'].iloc[-1]:%d %b %H:%M}")
            if rules["alarm"]:
                st.error(f"ALARM on {line}: the latest reading breaks a control chart rule.")
            elif rules["warning"]:
                st.warning(f"Drift warning on {line}: the readings are drifting to one side.")
            else:
                st.success(f"{line}: in control")
            st.plotly_chart(charts.control_chart(
                g["timestamp"].iloc[tail], g["value"].iloc[tail], s["rules"].iloc[tail],
                s["limits"], lsl, usl, f"{line}: {characteristic}", unit),
                key=f"live_chart_{line}")


monitor()
