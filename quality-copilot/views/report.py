"""Shift report: the local model writes a short report from the computed facts."""
import pandas as pd
import streamlit as st

import ui
from ai.report_facts import build_facts
from ai.report_writer import LLMUnavailable, write_report
from show_report_facts import load

st.title("Shift report")
st.write("The local language model (via Ollama) writes a short plain-language report for each "
         "line from the computed numbers of the last 7 days. Every number, date and time in the "
         "text is checked automatically against those numbers. Writing takes about 2 minutes.")

end = ui.latest_time()
if end is None:
    st.info("There is no data yet. Add records on the Data entry page or run the simulator.")
    st.stop()

if st.button("Write report", type="primary"):
    meas, specs = load("measurements"), load("specs")
    prod, down, defects = load("production"), load("downtime"), load("defects")
    if len(meas) == 0:
        st.info("There are no readings to write a report about.")
        st.stop()
    start = end - pd.Timedelta(days=7)
    facts = [build_facts(line, "Last 7 days", start, end, meas, specs, prod, down, defects)
             for line in sorted(meas["line"].unique())]
    with st.spinner("The model is writing the report, please wait about 2 minutes ..."):
        try:
            st.session_state["report"] = write_report(facts)
        except LLMUnavailable as error:
            st.session_state.pop("report", None)
            st.error(str(error))

result = st.session_state.get("report")
if result:
    st.write(result["text"])
    if result["ok"]:
        st.success("Number check passed: every number, date and time in the text was found "
                   "in the computed facts.")
    else:
        st.warning("Number check FAILED, so this report is not verified. Numbers that are not "
                   "in the facts: " + ", ".join(result["bad_numbers"]))
    st.caption(f"Model {result['model']}, {result['seconds']} seconds, "
               f"attempts: {result['attempts']}. Text written by an AI model from computed "
               "numbers. Demo data is simulated.")
