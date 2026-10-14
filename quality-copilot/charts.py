"""Plotly charts used by the app pages."""
import numpy as np
import plotly.graph_objects as go

RED, AMBER, BLUE, GREY, GREEN = "#d62728", "#ff9f1c", "#1f77b4", "#7f7f7f", "#2ca02c"
LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))


def control_chart(ts, values, rules, limits, lsl, usl, title, unit=""):
    """Readings over time with control limits, spec limits, alarms and drift warnings."""
    values = np.asarray(values, dtype=float)
    ts = ts.reset_index(drop=True)
    alarm = rules["alarm"].to_numpy()
    warning = rules["warning"].to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=values, mode="lines+markers", name="Reading",
                             line=dict(color=BLUE, width=1), marker=dict(size=4),
                             hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f} " + unit + "<extra></extra>"))

    def flagged(mask, name, color, symbol, size):
        if mask.any():
            labels = []
            for i in np.flatnonzero(mask):
                hit = [str(k) for k in (1, 2, 3, 4) if rules[f"rule{k}"].iloc[i]]
                labels.append("rule " + ", ".join(hit))
            fig.add_trace(go.Scatter(x=ts[mask], y=values[mask], mode="markers", name=name,
                                     marker=dict(color=color, size=size, symbol=symbol),
                                     text=labels,
                                     hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f}<br>%{text}<extra></extra>"))

    flagged(warning, "Drift warning", AMBER, "diamond", 8)
    flagged(alarm, "Alarm", RED, "circle", 10)

    lines = [(limits["center"], "Centre", GREY, "solid"),
             (limits["ucl"], "UCL", GREEN, "dash"), (limits["lcl"], "LCL", GREEN, "dash")]
    if usl is not None:
        lines.append((usl, "USL", RED, "dot"))
    if lsl is not None:
        lines.append((lsl, "LSL", RED, "dot"))
    for y, name, color, dash in lines:
        fig.add_hline(y=y, line=dict(color=color, dash=dash, width=1.2),
                      annotation_text=f"{name} {y:.2f}", annotation_position="right",
                      annotation_font_size=10)
    fig.update_layout(title=title, yaxis_title=unit or None, **{**LAYOUT, "margin": dict(l=10, r=110, t=40, b=10)})
    return fig


def pareto_chart(par, title="Pareto of scrap by defect type"):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=par.iloc[:, 0], y=par["quantity"], name="Scrap (m)",
                         marker_color=[BLUE if v else GREY for v in par["vital_few"]]))
    fig.add_trace(go.Scatter(x=par.iloc[:, 0], y=par["cumulative_percent"], name="Cumulative %",
                             yaxis="y2", mode="lines+markers", line=dict(color=RED)))
    fig.add_hline(y=80, yref="y2", line=dict(color=RED, dash="dot", width=1))
    fig.update_layout(title=title, yaxis=dict(title="Scrap (m)"),
                      yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                                  range=[0, 105]), **{**LAYOUT, "margin": dict(l=10, r=60, t=40, b=10)})
    return fig


def oee_bars(values, benchmark):
    names = ["Availability", "Performance", "Quality", "OEE"]
    keys = ["availability", "performance", "quality", "oee"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[values[k] * 100 for k in keys], name="Actual",
                         marker_color=BLUE, text=[f"{values[k]*100:.1f}%" for k in keys],
                         textposition="outside"))
    fig.add_trace(go.Scatter(x=names, y=[benchmark[k] * 100 for k in keys], name="Reference",
                             mode="markers", marker=dict(color=RED, symbol="line-ew", size=26,
                                                         line=dict(width=3, color=RED))))
    fig.update_layout(title="OEE and its three factors", yaxis=dict(range=[0, 110], title="%"),
                      **LAYOUT)
    return fig


def loss_chart(losses):
    colors = {"Availability": BLUE, "Performance": AMBER, "Quality": RED}
    data = losses.sort_values("minutes")
    fig = go.Figure(go.Bar(x=data["minutes"], y=data["loss"], orientation="h",
                           marker_color=[colors.get(t, GREY) for t in data["type"]],
                           text=[f"{p:.1f}%" for p in data["percent_of_planned"]],
                           textposition="outside",
                           hovertemplate="%{y}<br>%{x:.0f} minutes<extra></extra>"))
    fig.update_layout(title="Where the planned time was lost (minutes; label = % of planned)",
                      xaxis_title="Minutes", **LAYOUT)
    return fig


def trend_chart(df, x, y, title, benchmark=None):
    fig = go.Figure(go.Scatter(x=df[x], y=df[y] * 100, mode="lines+markers",
                               line=dict(color=BLUE), name=y.title()))
    if benchmark is not None:
        fig.add_hline(y=benchmark * 100, line=dict(color=RED, dash="dot", width=1),
                      annotation_text="reference", annotation_position="right")
    fig.update_layout(title=title, yaxis_title="%", **{**LAYOUT, "margin": dict(l=10, r=80, t=40, b=10)})
    return fig


def bars(df, x, y, title, ytitle):
    fig = go.Figure(go.Bar(x=df[x], y=df[y], marker_color=BLUE,
                           text=[f"{v:.1f}" for v in df[y]], textposition="outside"))
    fig.update_layout(title=title, yaxis_title=ytitle, **LAYOUT)
    return fig


def mix_heatmap(table):
    fig = go.Figure(go.Heatmap(z=table.to_numpy(), x=list(table.columns), y=list(table.index),
                               text=table.to_numpy(), texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(title="Defect events by type and shift", **LAYOUT)
    return fig
