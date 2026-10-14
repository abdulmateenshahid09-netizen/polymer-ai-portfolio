"""Quality Analytics Copilot. Start it with:  streamlit run quality-copilot\\app.py"""
import streamlit as st

import ui

st.set_page_config(page_title="Quality Analytics Copilot", layout="wide")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/entry.py", title="Data entry", icon=":material/edit_note:"),
    st.Page("views/spc.py", title="SPC and capability", icon=":material/monitoring:"),
    st.Page("views/live_monitor.py", title="Live monitor", icon=":material/sensors:"),
    st.Page("views/oee.py", title="OEE", icon=":material/precision_manufacturing:"),
    st.Page("views/defects.py", title="Defects", icon=":material/bug_report:"),
]
page = st.navigation(pages)
with st.sidebar:
    ui.sidebar_filters()
page.run()
