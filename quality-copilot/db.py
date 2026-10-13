import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "quality.db"

TABLES = ["measurements", "specs", "production", "downtime", "defects", "process_log"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    line TEXT NOT NULL,
    product TEXT NOT NULL,
    characteristic TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    shift TEXT,
    operator TEXT,
    lot TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS specs (
    product TEXT NOT NULL,
    characteristic TEXT NOT NULL,
    lsl REAL,
    usl REAL,
    target REAL,
    unit TEXT,
    PRIMARY KEY (product, characteristic)
);

CREATE TABLE IF NOT EXISTS production (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    line TEXT NOT NULL,
    product TEXT NOT NULL,
    planned_minutes REAL NOT NULL,
    run_minutes REAL NOT NULL,
    ideal_rate_per_min REAL NOT NULL,
    total_count REAL NOT NULL,
    good_count REAL NOT NULL,
    shift TEXT
);

CREATE TABLE IF NOT EXISTS downtime (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    line TEXT NOT NULL,
    category TEXT NOT NULL,
    reason TEXT,
    minutes REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    line TEXT NOT NULL,
    product TEXT NOT NULL,
    defect_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    shift TEXT,
    lot TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS process_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    line TEXT NOT NULL,
    screw_rpm REAL,
    haul_speed_mpm REAL,
    melt_temp_c REAL,
    mfr REAL,
    lot TEXT
);
"""


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def read_table(name):
    if name not in TABLES:
        raise ValueError(f"Unknown table: {name}")
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
    conn.close()
    return df


if __name__ == "__main__":
    init_db()
    print(f"Database ready at: {DB_PATH}")
    for t in TABLES:
        print(f"{t:14s} rows: {len(read_table(t))}")
