"""Import your own records from CSV or Excel files into the database."""
import sys
from pathlib import Path

import pandas as pd

from db import get_connection, init_db, DB_PATH

# required and optional columns for each table you can import into
COLUMNS = {
    "measurements": (
        ["timestamp", "line", "product", "characteristic", "value"],
        ["unit", "shift", "operator", "lot", "notes"],
    ),
    "production": (
        ["start_time", "end_time", "line", "product", "planned_minutes", "run_minutes",
         "ideal_rate_per_min", "total_count", "good_count"],
        ["shift"],
    ),
    "downtime": (
        ["start_time", "end_time", "line", "category", "minutes"],
        ["reason"],
    ),
    "defects": (
        ["timestamp", "line", "product", "defect_type", "quantity"],
        ["shift", "lot", "notes"],
    ),
    "process_log": (
        ["timestamp", "line"],
        ["screw_rpm", "haul_speed_mpm", "melt_temp_c", "mfr", "lot"],
    ),
    "specs": (
        ["product", "characteristic", "lsl", "usl"],
        ["target", "unit"],
    ),
}

TIME_COLUMNS = ["timestamp", "start_time", "end_time"]
NUMBER_COLUMNS = ["value", "planned_minutes", "run_minutes", "ideal_rate_per_min",
                  "total_count", "good_count", "minutes", "quantity", "screw_rpm",
                  "haul_speed_mpm", "melt_temp_c", "mfr", "lsl", "usl", "target"]
# columns that identify a record, used to skip records that already exist
KEYS = {
    "measurements": ["timestamp", "line", "characteristic"],
    "production": ["start_time", "line"],
    "downtime": ["start_time", "line", "category"],
    "defects": ["timestamp", "line", "defect_type"],
    "process_log": ["timestamp", "line"],
}

EXAMPLES = {
    "measurements": ["2026-10-12 08:00:00", "Line 1", "PE100 SDR11 110mm", "wall_thickness",
                     10.45, "mm", "Day", "Operator A", "L1-001", ""],
    "production": ["2026-10-12 06:00:00", "2026-10-12 14:00:00", "Line 1", "PE100 SDR11 110mm",
                   480, 430, 1.4, 520, 505, "Day"],
    "downtime": ["2026-10-12 09:00:00", "2026-10-12 09:30:00", "Line 1", "Changeover", 30,
                 "Size change"],
    "defects": ["2026-10-12 10:15:00", "Line 1", "PE100 SDR11 110mm", "Surface scratches",
                12.5, "Day", "L1-001", ""],
    "process_log": ["2026-10-12 08:00:00", "Line 1", 100, 1.24, 210, 0.30, "L1-001"],
    "specs": ["PE100 SDR11 110mm", "wall_thickness", 10.0, 11.1, 10.4, "mm"],
}


def shift_of(ts):
    h = ts.hour
    if 6 <= h < 14:
        return "Day"
    if 14 <= h < 22:
        return "Evening"
    return "Night"


def _tidy_columns(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def read_file(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Use a .csv or .xlsx file")
    return _tidy_columns(df)


def read_upload(uploaded):
    """Read a file uploaded in the app (it has a .name and can be read like a file)."""
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        raise ValueError("Use a .csv or .xlsx file")
    return _tidy_columns(df)


def validate(df, table):
    """Return (clean rows, list of problems). Bad rows are skipped, not guessed."""
    required, optional = COLUMNS[table]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df[[c for c in required + optional if c in df.columns]].copy()
    problems, bad = [], pd.Series(False, index=df.index)

    def flag(mask, why):
        nonlocal bad
        for i in df.index[mask & ~bad]:
            problems.append(f"row {i + 2}: {why}")
        bad = bad | mask

    for c in required:
        flag(df[c].isna() | (df[c].astype(str).str.strip() == ""), f"{c} is empty")

    for c in [c for c in TIME_COLUMNS if c in df.columns]:
        parsed = pd.to_datetime(df[c], errors="coerce")
        flag(parsed.isna() & ~bad, f"{c} is not a valid date and time")
        df[c] = parsed

    for c in [c for c in NUMBER_COLUMNS if c in df.columns]:
        num = pd.to_numeric(df[c], errors="coerce")
        if c in required:
            flag(num.isna() & ~bad, f"{c} is not a number")
        df[c] = num

    for c in ["planned_minutes", "run_minutes", "total_count", "good_count",
              "minutes", "quantity"]:
        if c in df.columns:
            flag((df[c] < 0) & ~bad, f"{c} is negative")
    if table == "production":
        flag((df["run_minutes"] > df["planned_minutes"]) & ~bad,
             "run_minutes is larger than planned_minutes")
        flag((df["good_count"] > df["total_count"]) & ~bad,
             "good_count is larger than total_count")
    if table == "specs":
        flag((df["lsl"] >= df["usl"]) & ~bad, "lsl must be smaller than usl")

    df = df[~bad].copy()
    if table in ("measurements", "defects", "production"):
        if "shift" not in df.columns:
            df["shift"] = None
        when = df["start_time"] if table == "production" else df["timestamp"]
        df["shift"] = df["shift"].where(df["shift"].notna(), pd.to_datetime(when).apply(shift_of))
    for c in TIME_COLUMNS:
        if c in df.columns:
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    problems.sort(key=lambda p: int(p.split()[1].rstrip(":")))
    return df, problems


def import_file(path, table):
    return import_dataframe(read_file(path), table)


def import_dataframe(raw, table):
    """Validate and store the rows of a table of records. Returns a summary dict."""
    if table not in COLUMNS:
        raise ValueError(f"Table must be one of: {', '.join(COLUMNS)}")
    init_db()
    df, problems = validate(raw, table)
    total_rows = len(df) + len(problems)

    conn = get_connection()
    skipped = 0
    if table == "specs":
        conn.executemany(
            "INSERT OR REPLACE INTO specs (product, characteristic, lsl, usl, target, unit) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(r["product"], r["characteristic"], r["lsl"], r["usl"],
              r.get("target"), r.get("unit")) for _, r in df.iterrows()])
    else:
        before = len(df)
        df = df.drop_duplicates(subset=KEYS[table])
        existing = pd.read_sql_query(f"SELECT {', '.join(KEYS[table])} FROM {table}", conn)
        df = df.merge(existing.drop_duplicates(), on=KEYS[table], how="left", indicator=True)
        df = df[df["_merge"] == "left_only"].drop(columns="_merge")
        skipped = before - len(df)
        df.to_sql(table, conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return {"total_rows": total_rows, "imported": len(df), "already_there": skipped,
            "rejected": len(problems), "problems": problems}


def make_templates():
    folder = DB_PATH.parent / "templates"
    folder.mkdir(parents=True, exist_ok=True)
    for table, (required, optional) in COLUMNS.items():
        cols = required + optional
        pd.DataFrame([EXAMPLES[table]], columns=cols).to_csv(
            folder / f"{table}_template.csv", index=False)
    return folder


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "templates":
        print(f"Templates written to: {make_templates()}")
    elif len(sys.argv) == 3:
        result = import_file(sys.argv[1], sys.argv[2])
        print(f"Rows in file: {result['total_rows']}   Imported: {result['imported']}   "
              f"Already in database: {result['already_there']}   Rejected: {result['rejected']}")
        for p in result["problems"][:20]:
            print("  " + p)
        if result["rejected"] > 20:
            print(f"  ... and {result['rejected'] - 20} more")
    else:
        print("Usage:")
        print("  python quality-copilot\\importer.py templates")
        print("  python quality-copilot\\importer.py <file.csv or file.xlsx> <table>")
