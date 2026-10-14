"""Print the facts that the report writer may use, for each line and two periods."""
import json

import pandas as pd

import ui
from ai.report_facts import build_facts
from db import read_table

TIME_COLUMN = {"measurements": "timestamp", "defects": "timestamp",
               "production": "start_time", "downtime": "start_time"}


def load(name):
    df = read_table(name)
    if name in TIME_COLUMN and len(df):
        df[TIME_COLUMN[name]] = pd.to_datetime(df[TIME_COLUMN[name]])
    return df


def main():
    meas, specs = load("measurements"), load("specs")
    prod, down, defects = load("production"), load("downtime"), load("defects")
    end = ui.latest_time()   # the dashboard measures "last 7 days" from here
    for label, days in (("All data", None), ("Last 7 days", 7)):
        start = None if days is None else end - pd.Timedelta(days=days)
        for line in sorted(meas["line"].unique()):
            facts = build_facts(line, label, start, end, meas, specs, prod, down, defects)
            print(json.dumps(facts, indent=1, default=str))
            print()


if __name__ == "__main__":
    main()
