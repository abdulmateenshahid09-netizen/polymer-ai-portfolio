"""OEE: availability, performance, quality, and where the planned time was lost."""
import pandas as pd
import streamlit as st

import charts
import ui
from metrics.oee import BENCHMARK, calculate_oee, loss_breakdown, oee_by

st.title("OEE")
filters = ui.get_filters()
prod = ui.apply(ui.table("production"), filters, "start_time")
down = ui.apply(ui.table("downtime"), filters, "start_time")
ui.need(prod, "No production runs in this period. Add them on the Data entry page "
              "(Production run), or choose a longer period.")

result = calculate_oee(prod)
k1, k2, k3, k4 = st.columns(4)
for column, key, label in ((k1, "availability", "Availability"), (k2, "performance", "Performance"),
                           (k3, "quality", "Quality"), (k4, "oee", "OEE")):
    gap = (result[key] - BENCHMARK[key]) * 100
    column.metric(label, f"{result[key] * 100:.1f}%", f"{gap:+.1f} points vs reference",
                  delta_color="normal")
st.caption(f"{result['records']} production runs, {result['planned_minutes']:.0f} planned minutes. "
           "The reference levels (90%, 95%, 99%, OEE 85%) are commonly quoted 'world class' values.")
for text in result["warnings"]:
    st.warning(text)

left, right = st.columns(2)
left.plotly_chart(charts.oee_bars(result, BENCHMARK))
losses = loss_breakdown(prod, down)
right.plotly_chart(charts.loss_chart(losses["losses"]))
st.write(f"Fully productive time: **{losses['productive_minutes']:.0f} of "
         f"{losses['planned_minutes']:.0f} planned minutes ({losses['productive_percent']:.1f}%)**. "
         "The rest was lost to the items in the chart.")

daily = oee_by(prod, "date")
if len(daily) > 1:
    st.plotly_chart(charts.trend_chart(daily, "date", "oee", "OEE by day", BENCHMARK["oee"]))

st.subheader("By line and by shift (%)")
a, b = st.columns(2)
for column, by in ((a, "line"), (b, "shift")):
    table = oee_by(prod, by)
    for name in ("availability", "performance", "quality", "oee"):
        table[name] = (table[name] * 100).round(1)
    column.dataframe(table, hide_index=True)
with st.expander("Loss table"):
    st.dataframe(losses["losses"], hide_index=True)
