"""Write the report for the last 7 days with the local model and show the number check."""
import pandas as pd

import ui
from ai.report_facts import build_facts
from ai.report_writer import MODEL, LLMUnavailable, write_report
from show_report_facts import load


def main():
    meas, specs = load("measurements"), load("specs")
    prod, down, defects = load("production"), load("downtime"), load("defects")
    end = ui.latest_time()
    start = end - pd.Timedelta(days=7)
    facts = [build_facts(line, "Last 7 days", start, end, meas, specs, prod, down, defects)
             for line in sorted(meas["line"].unique())]
    print(f"Asking {MODEL} for the report (about a minute) ...\n")
    try:
        r = write_report(facts)
    except LLMUnavailable as error:
        print(error)
        return
    print(r["text"])
    print()
    print("Numbers verified:", r["ok"], "| attempts:", r["attempts"], "| seconds:", r["seconds"])
    if not r["ok"]:
        print("Numbers that are not in the facts:", ", ".join(r["bad_numbers"]))


if __name__ == "__main__":
    main()
