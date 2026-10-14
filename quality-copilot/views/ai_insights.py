"""AI insights: anomalies next to SPC alarms, and the off-target risk model against simple rules."""
import numpy as np
import pandas as pd
import streamlit as st

import charts
import ui
from ai.anomaly import anomaly_events, detect_anomalies
from ai.risk_model import (FEATURES, HORIZON, K_SIGMA, SETTINGS, build_dataset, compare_verdict,
                           feature_importance, naive_scores, score_report, time_split, train_model)
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


# ---------------------------------------------------------------- 2. off-target risk
@st.cache_data(show_spinner="Training the risk model ...")
def risk_analysis(wall_df, process_df):
    """Train on the first 20 days, test on the rest, and compare with the simple rules."""
    per_line = {}
    for line, g in wall_df.groupby("line"):
        d = build_dataset(g[["timestamp", "value", "shift"]],
                          process_df[process_df["line"] == line])
        d["line"] = line
        per_line[line] = d
    if not per_line:
        raise ValueError("No line has wall thickness readings together with process settings")
    data = pd.concat(per_line.values(), ignore_index=True)
    train, test = time_split(data)
    if len(test) == 0:
        raise ValueError("Not enough days of data for a test period after the first 20 days")
    model = train_model(train)
    proba = model.predict_proba(test[FEATURES])[:, 1]

    rows = []
    for name, scores in [("XGBoost model", proba)] + list(naive_scores(test).items()):
        r = score_report(test["y"], scores)
        rows.append({"method": name, "AUC": r["auc"], "avg precision": r["average_precision"],
                     "precision (top 10%)": r["precision_top"],
                     "recall (top 10%)": r["recall_top"]})
    table = pd.DataFrame(rows)

    threshold = float(np.quantile(proba, 0.9))
    top = test[proba >= threshold]
    latest = []
    for line, d in per_line.items():
        last = d.sort_values("timestamp").iloc[[-1]]
        score = float(model.predict_proba(last[FEATURES])[:, 1][0])
        latest.append({"line": line, "time": last["timestamp"].iloc[0],
                       "risk score": round(score, 2),
                       "level": "HIGH" if score >= threshold else "normal",
                       "naive |z|": round(float(abs(last["z"].iloc[0])), 2)})
    return {"table": table, "n_train": int(len(train)), "pos_train": int(train["y"].sum()),
            "n_test": int(len(test)), "pos_test": int(test["y"].sum()),
            "base_rate": float(test["y"].mean()), "latest": pd.DataFrame(latest),
            "importance": feature_importance(model).head(5), "threshold": threshold,
            "night_top": float(top["shift_night"].mean()), "night_all": float(test["shift_night"].mean())}


st.header("2. Off-target risk: model against simple rules")
st.write(f"The model estimates whether any of the next {HORIZON} readings (2 hours) will be more "
         f"than {K_SIGMA:g} baseline sigma away from normal. It is trained on the first 20 days "
         "and judged only on the last 10 days it has never seen.")
st.caption("This comparison always uses all lines and the last 10 days. The sidebar period does "
           "not change it.")

process = ui.table("process_log")
if len(process) == 0:
    st.info("No process settings are stored yet (screw rpm, haul-off speed, melt temperature, "
            "MFR). The risk model needs them. Load the demo data to see this section.")
else:
    try:
        risk = risk_analysis(wall[["timestamp", "line", "value", "shift"]],
                             process[["timestamp", "line"] + SETTINGS])
    except ValueError as error:
        st.info(f"The risk model cannot run yet: {error}.")
        risk = None
    if risk is not None:
        beats, text = compare_verdict(risk["table"])
        if beats is None:
            st.info(text)
        elif beats:
            st.success(text)
        else:
            st.warning(text)

        c1, c2, c3 = st.columns(3)
        c1.metric("Training readings", risk["n_train"], f"{risk['pos_train']} off-target ahead",
                  delta_color="off", delta_arrow="off")
        c2.metric("Test readings", risk["n_test"], f"{risk['pos_test']} off-target ahead",
                  delta_color="off", delta_arrow="off")
        c3.metric("Base rate", f"{risk['base_rate'] * 100:.1f}%", "readings followed by a miss",
                  delta_color="off", delta_arrow="off")

        shown_table = risk["table"].copy()
        for column in shown_table.columns[1:]:
            shown_table[column] = shown_table[column].round(3)
        st.dataframe(shown_table, hide_index=True)
        st.caption("AUC: 0.5 is a coin flip, 1.0 is perfect. Top 10%: only the riskiest tenth of "
                   "readings is flagged.")

        st.subheader("Latest risk scores")
        latest = risk["latest"]
        if filters["line"]:
            latest = latest[latest["line"] == filters["line"]]
        st.dataframe(latest.sort_values("line"), hide_index=True)
        st.caption(f"The score is a ranking, not a probability. HIGH means it is in the riskiest "
                   f"tenth of the test period (score {risk['threshold']:.2f} or more).")

        with st.expander("Why the model is not better, and what it leaned on"):
            st.write("Only a few readings are actually out of specification, and the simulated "
                     "faults start abruptly, so there is no early warning to learn. A model "
                     "cannot predict what gives no precursor.")
            st.write(f"Night shift matters: {risk['night_top'] * 100:.0f}% of the readings the model "
                     f"calls HIGH are night shift, against {risk['night_all'] * 100:.0f}% of all test "
                     "readings. A HIGH score therefore often means 'it is night', so check the "
                     "chart before acting on it.")
            st.write("What the model leaned on most (share of its decisions):")
            st.dataframe(risk["importance"].round(3).rename("importance").reset_index()
                         .rename(columns={"index": "input"}), hide_index=True)