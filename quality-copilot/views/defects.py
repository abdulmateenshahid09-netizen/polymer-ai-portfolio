"""Defects: Pareto, scrap rates, defect mix by shift, and unusual resin lots."""
import streamlit as st

import charts
import ui
from metrics.defects import (defect_mix, lot_defect_outliers, lot_summary, pareto,
                             scrap_rate)

st.title("Defects and scrap")
filters = ui.get_filters()
defects = ui.apply(ui.table("defects"), filters, "timestamp")
prod = ui.apply(ui.table("production"), filters, "start_time")
log = ui.apply(ui.table("process_log"), filters, "timestamp")
ui.need(defects, "No defects in this period. Add them on the Data entry page (Defect), "
                 "or choose a longer period.")

k1, k2, k3 = st.columns(3)
k1.metric("Defect records", len(defects))
k2.metric("Scrapped length", f"{defects['quantity'].sum():.0f} m")
if len(prod):
    k3.metric("Scrap rate", f"{defects['quantity'].sum() / prod['total_count'].sum() * 100:.2f}%")

par = pareto(defects)
left, right = st.columns([3, 2])
left.plotly_chart(charts.pareto_chart(par))
right.dataframe(par, hide_index=True)
vital = ", ".join(par.loc[par["vital_few"], "defect_type"])
st.write(f"The **vital few** (causes needed to reach 80% of the scrap): {vital}.")

if len(prod):
    a, b = st.columns(2)
    by_line = scrap_rate(defects, prod, "line")
    by_shift = scrap_rate(defects, prod, "shift")
    a.plotly_chart(charts.bars(by_line, "line", "scrap_percent", "Scrap rate by line", "% of output"))
    b.plotly_chart(charts.bars(by_shift, "shift", "scrap_percent", "Scrap rate by shift", "% of output"))
else:
    st.info("Add production runs to see scrap rates (scrap as a share of output).")

st.subheader("Do the causes differ between shifts?")
try:
    mix = defect_mix(defects, "defect_type", "shift")
except ValueError:
    st.info("At least two defect types and two shifts are needed for this comparison.")
else:
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.mix_heatmap(mix["table"]))
    verdict = ("The mix of causes differs between shifts." if mix["different"]
               else "No clear difference between shifts.")
    c2.write(f"**{verdict}**")
    c2.write(f"Chi-square p-value {mix['p_value']:.4f}; effect size (Cramer's V) {mix['cramers_v']}. "
             "A p-value below 0.05 means the difference is unlikely to be chance.")

st.subheader("Resin lots")
if len(log) == 0:
    st.info("Lot comparison needs process log records (resin lot for each reading time).")
else:
    lots = lot_summary(defects, log)
    st.write("Scrap per 100 running hours for each lot. A lot is flagged when it is far above the others.")
    st.dataframe(lots.head(10), hide_index=True)
    odd = lot_defect_outliers(defects, log)
    if len(odd):
        st.write("Lots unusually bad for one specific defect type:")
        st.dataframe(odd, hide_index=True)
        st.caption("Check whether the lot, or an equipment fault during that lot, is the real cause.")
    else:
        st.write("No lot stands out for a specific defect type.")
