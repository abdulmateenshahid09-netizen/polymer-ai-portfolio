"""Data entry: forms for readings, production runs, downtime, defects, limits, and file import."""
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import importer
import store
import ui

TYPICAL_DOWNTIME = ["Changeover", "Material shortage", "Mechanical breakdown",
                    "Die cleaning", "Quality hold", "Minor stoppage"]
TYPICAL_DEFECTS = ["Surface scratches", "Black specks", "Ovality", "Marking error",
                   "Wall thickness out of spec"]


def pick_time(label, key, hours_back=0):
    """A date box and a time box, starting at the current time."""
    start = (datetime.now() - timedelta(hours=hours_back)).replace(second=0, microsecond=0)
    c1, c2 = st.columns(2)
    day = c1.date_input(f"{label} date", value=start.date(), key=f"{key}_date")
    clock = c2.time_input(f"{label} time", value=start.time(), step=60, key=f"{key}_time")
    return datetime.combine(day, clock)


def choose(label, options, key, first=True, default=None):
    """A dropdown of known values that also lets you type a new one.
    It starts on `default` (for example the value used last time) when it is in the list."""
    options = [o for o in options if o]
    index = None
    if options and first:
        index = options.index(default) if default in options else 0
    return st.selectbox(label, options, key=key, accept_new_options=True,
                        index=index, placeholder="Choose or type")


def report(result):
    ok, message = result
    (st.success if ok else st.error)(message)


def recent(table, columns, sort_column, n=8):
    df = ui.table(table)
    if len(df) == 0:
        st.caption("Nothing saved yet.")
        return
    st.dataframe(df.sort_values(sort_column, ascending=False).head(n)[columns], hide_index=True)


st.title("Data entry")
st.caption("Type in a record and save it. The other pages use the new record straight away.")
tabs = st.tabs(["Measurement", "Production run", "Downtime", "Defect", "Specifications", "Import file"])

specs = ui.table("specs")
spec_products = list(specs["product"]) if len(specs) else []
products = sorted(set(spec_products) | set(ui.distinct("measurements", "product"))
                  | set(ui.distinct("production", "product")))
lines = ui.all_lines() or ["Line 1"]

# ---- Measurement
with tabs[0]:
    product = choose("Product", products, "m_product",
                     default=ui.last_used("measurements", "product"))
    chars = sorted(specs.loc[specs["product"] == product, "characteristic"]) if len(specs) else []
    characteristic = choose("Characteristic", chars, "m_char",
                            default=ui.last_used("measurements", "characteristic"))
    spec = store.get_spec(product, characteristic) if product and characteristic else None
    if spec and spec["lsl"] is not None and spec["usl"] is not None:
        st.caption(f"Limits: {spec['lsl']} to {spec['usl']} {spec['unit'] or ''}"
                   + (f", target {spec['target']}" if spec["target"] is not None else ""))
    with st.form("measurement_form"):
        line = choose("Line", lines, "m_line", default=ui.last_used("measurements", "line"))
        when = pick_time("Reading", "m")
        default = float(spec["target"]) if spec and spec["target"] is not None else 0.0
        value = st.number_input("Value", value=default, step=0.01, format="%.3f")
        operator = choose("Operator (optional)", ui.distinct("measurements", "operator"),
                          "m_operator", first=False)
        lot = choose("Resin lot (optional)", ui.distinct("measurements", "lot"), "m_lot", first=False)
        notes = st.text_input("Notes (optional)")
        confirm = st.checkbox("The value is correct even though it is far outside the limits")
        if st.form_submit_button("Save reading", type="primary"):
            report(store.add_measurement(when, line, product, characteristic, value,
                                         operator, lot, notes, confirm_outlier=confirm))
    st.subheader("Latest readings")
    recent("measurements", ["timestamp", "line", "characteristic", "value", "shift", "lot"], "timestamp")

# ---- Production run
with tabs[1]:
    with st.form("production_form"):
        product_p = choose("Product", products, "p_product", default=ui.last_used("production", "product"))
        line_p = choose("Line", lines, "p_line", default=ui.last_used("production", "line"))
        start = pick_time("Start", "p_start", hours_back=8)
        end = pick_time("End", "p_end")
        c1, c2 = st.columns(2)
        planned = c1.number_input("Planned minutes", value=480.0, min_value=0.0, step=10.0)
        run = c2.number_input("Run minutes (planned minus downtime)", value=480.0, min_value=0.0, step=10.0)
        c3, c4, c5 = st.columns(3)
        ideal = c3.number_input("Ideal rate (metres per minute)", value=1.40, min_value=0.0, step=0.05)
        total = c4.number_input("Total produced (m)", value=0.0, min_value=0.0, step=10.0)
        good = c5.number_input("Good output (m)", value=0.0, min_value=0.0, step=10.0)
        if st.form_submit_button("Save production run", type="primary"):
            report(store.add_production(start, end, line_p, product_p, planned, run, ideal, total, good))
    st.subheader("Latest production runs")
    recent("production", ["start_time", "line", "shift", "run_minutes", "total_count", "good_count"],
           "start_time")

# ---- Downtime
with tabs[2]:
    with st.form("downtime_form"):
        line_d = choose("Line", lines, "d_line", default=ui.last_used("downtime", "line"))
        start_d = pick_time("Start", "d_start", hours_back=1)
        end_d = pick_time("End", "d_end")
        categories = sorted(set(TYPICAL_DOWNTIME) | set(ui.distinct("downtime", "category")))
        category = choose("Category", categories, "d_category", first=False)
        reason = st.text_input("Reason (optional)")
        if st.form_submit_button("Save downtime", type="primary"):
            report(store.add_downtime(start_d, end_d, line_d, category, reason))
    st.subheader("Latest downtime")
    recent("downtime", ["start_time", "line", "category", "reason", "minutes"], "start_time")

# ---- Defect
with tabs[3]:
    with st.form("defect_form"):
        product_f = choose("Product", products, "f_product", default=ui.last_used("defects", "product"))
        line_f = choose("Line", lines, "f_line", default=ui.last_used("defects", "line"))
        when_f = pick_time("Found", "f")
        kinds = sorted(set(TYPICAL_DEFECTS) | set(ui.distinct("defects", "defect_type")))
        defect_type = choose("Defect type", kinds, "f_type", first=False)
        quantity = st.number_input("Scrapped length (m)", value=0.0, min_value=0.0, step=1.0)
        lot_f = choose("Resin lot (optional)", ui.distinct("defects", "lot"), "f_lot", first=False)
        notes_f = st.text_input("Notes (optional)", key="f_notes")
        if st.form_submit_button("Save defect", type="primary"):
            report(store.add_defect(when_f, line_f, product_f, defect_type, quantity, lot_f, notes_f))
    st.subheader("Latest defects")
    recent("defects", ["timestamp", "line", "defect_type", "quantity", "shift", "lot"], "timestamp")

# ---- Specifications
with tabs[4]:
    st.write("Specification limits are what Cp, Cpk and out-of-spec counts are measured against.")
    with st.form("spec_form"):
        product_s = choose("Product", products, "s_product", first=False)
        characteristic_s = st.text_input("Characteristic (for example wall_thickness)")
        c1, c2, c3 = st.columns(3)
        lsl = c1.number_input("Lower limit (LSL)", value=None, placeholder="optional", format="%.3f")
        usl = c2.number_input("Upper limit (USL)", value=None, placeholder="optional", format="%.3f")
        target = c3.number_input("Target", value=None, placeholder="optional", format="%.3f")
        unit = st.text_input("Unit (for example mm)")
        if st.form_submit_button("Save limits", type="primary"):
            report(store.add_spec(product_s, characteristic_s, lsl, usl, target, unit))
    st.subheader("Stored limits")
    st.dataframe(ui.table("specs"), hide_index=True)

# ---- Import file
with tabs[5]:
    st.write("Load many records at once from a CSV or Excel file. Bad rows are rejected with a "
             "reason; records that already exist are skipped.")
    kind = st.selectbox("What kind of records are in the file?", list(importer.COLUMNS))
    required, optional = importer.COLUMNS[kind]
    st.caption("Required columns: " + ", ".join(required))
    st.caption("Optional columns: " + (", ".join(optional) if optional else "none"))
    template = pd.DataFrame([importer.EXAMPLES[kind]], columns=required + optional)
    st.download_button("Download a blank template (CSV)", template.to_csv(index=False),
                       file_name=f"{kind}_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded is not None and st.button("Import file", type="primary"):
        try:
            result = importer.import_dataframe(importer.read_upload(uploaded), kind)
        except Exception as error:
            st.error(str(error))
        else:
            a, b, c, d = st.columns(4)
            a.metric("Rows in file", result["total_rows"])
            b.metric("Imported", result["imported"])
            c.metric("Already stored", result["already_there"])
            d.metric("Rejected", result["rejected"])
            for problem in result["problems"][:20]:
                st.warning(problem)
