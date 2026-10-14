"""Shared helpers for the app: loading data, the sidebar filters, and empty-state messages."""
import pandas as pd
import streamlit as st

from db import get_connection, init_db, read_table

PERIODS = {"Last 24 hours": 1, "Last 3 days": 3, "Last 7 days": 7,
           "Last 14 days": 14, "Last 30 days": 30, "All data": None}
ALL_LINES = "All lines"
TIME_COLUMN = {"measurements": "timestamp", "defects": "timestamp", "process_log": "timestamp",
               "production": "start_time", "downtime": "start_time"}


def table(name):
    """A table from the database, with its time column converted to real dates."""
    init_db()
    df = read_table(name)
    column = TIME_COLUMN.get(name)
    if column and len(df):
        df[column] = pd.to_datetime(df[column])
    return df


def distinct(name, column):
    """Sorted distinct values of a column, used to fill dropdown lists."""
    init_db()
    conn = get_connection()
    rows = conn.execute(f"SELECT DISTINCT {column} FROM {name} "
                        f"WHERE {column} IS NOT NULL ORDER BY {column}").fetchall()
    conn.close()
    return [r[0] for r in rows]


def last_used(name, column):
    """The value of a column in the newest record of a table, used as the form default."""
    init_db()
    conn = get_connection()
    order = TIME_COLUMN.get(name, "id")
    row = conn.execute(f"SELECT {column} FROM {name} WHERE {column} IS NOT NULL "
                       f"ORDER BY {order} DESC LIMIT 1").fetchone()
    conn.close()
    return row[0] if row else None


def all_lines():
    found = set()
    for name in ("measurements", "production", "defects", "downtime"):
        found.update(distinct(name, "line"))
    return sorted(found)


def latest_time():
    """The newest record in the database (the app measures 'last 7 days' from here)."""
    init_db()
    conn = get_connection()
    stamps = []
    for name, column in (("measurements", "timestamp"), ("production", "end_time"),
                         ("defects", "timestamp"), ("downtime", "end_time")):
        value = conn.execute(f"SELECT MAX({column}) FROM {name}").fetchone()[0]
        if value:
            stamps.append(pd.Timestamp(value))
    conn.close()
    return max(stamps) if stamps else None


def sidebar_filters():
    """Draw the line and period filters. Called once from app.py so they persist."""
    st.header("Filters")
    st.selectbox("Line", [ALL_LINES] + all_lines(), key="flt_line")
    st.selectbox("Period", list(PERIODS), index=2, key="flt_period")
    end = latest_time()
    if end is not None:
        st.caption(f"Latest record: {end:%Y-%m-%d %H:%M}")


def get_filters():
    """The current filter choices: line (None = all), period label, start and end times."""
    line = st.session_state.get("flt_line", ALL_LINES)
    period = st.session_state.get("flt_period", "Last 7 days")
    end = latest_time()
    days = PERIODS.get(period)
    start = None if (days is None or end is None) else end - pd.Timedelta(days=days)
    return {"line": None if line == ALL_LINES else line, "period": period,
            "start": start, "end": end}


def apply(df, filters, time_column=None, line_column="line"):
    """Keep only the rows inside the chosen line and period."""
    if len(df) == 0:
        return df
    if filters["line"] and line_column in df.columns:
        df = df[df[line_column] == filters["line"]]
    if filters["start"] is not None and time_column:
        df = df[df[time_column] >= filters["start"]]
    return df


def need(df, message):
    """Show a friendly message and stop the page when there is nothing to show."""
    if len(df) == 0:
        st.info(message)
        st.stop()
