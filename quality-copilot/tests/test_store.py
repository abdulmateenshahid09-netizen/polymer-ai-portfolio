"""Tests for saving single entries. Each test uses its own empty temporary database."""
import pandas as pd
import pytest

import db
import importer
import store


@pytest.fixture(autouse=True)
def empty_database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    store.add_spec("P1", "wall", 10.0, 11.1, 10.4, "mm")


def stored(table):
    return db.read_table(table)


def test_shift_boundaries():
    assert store.shift_of("2026-01-01 05:59:00") == "Night"
    assert store.shift_of("2026-01-01 06:00:00") == "Day"
    assert store.shift_of("2026-01-01 13:59:00") == "Day"
    assert store.shift_of("2026-01-01 14:00:00") == "Evening"
    assert store.shift_of("2026-01-01 21:59:00") == "Evening"
    assert store.shift_of("2026-01-01 22:00:00") == "Night"


def test_measurement_within_spec_is_saved_with_shift_and_unit():
    ok, msg = store.add_measurement("2026-01-01 08:00", "Line 1", "P1", "wall", 10.5)
    assert ok and "within specification" in msg
    row = stored("measurements").iloc[0]
    assert row["value"] == 10.5 and row["shift"] == "Day" and row["unit"] == "mm"
    assert row["timestamp"] == "2026-01-01 08:00:00"


def test_measurement_out_of_spec_is_saved_and_reported():
    ok, msg = store.add_measurement("2026-01-01 08:00", "Line 1", "P1", "wall", 9.9)
    assert ok and "OUT OF SPEC: below" in msg
    ok, msg = store.add_measurement("2026-01-01 08:30", "Line 1", "P1", "wall", 11.3)
    assert ok and "OUT OF SPEC: above" in msg


def test_implausible_value_needs_confirmation():
    ok, msg = store.add_measurement("2026-01-01 08:00", "Line 1", "P1", "wall", 104.5)
    assert not ok and "typing mistake" in msg
    assert len(stored("measurements")) == 0
    ok, _ = store.add_measurement("2026-01-01 08:00", "Line 1", "P1", "wall", 104.5,
                                  confirm_outlier=True)
    assert ok and len(stored("measurements")) == 1


def test_measurement_rejects_duplicates_and_bad_input():
    assert store.add_measurement("2026-01-01 08:00", "Line 1", "P1", "wall", 10.5)[0]
    assert not store.add_measurement("2026-01-01 08:00", "Line 1", "P1", "wall", 10.6)[0]
    assert not store.add_measurement("2026-01-01 09:00", "Line 1", "P1", "wall", "abc")[0]
    assert not store.add_measurement("2026-01-01 09:00", "", "P1", "wall", 10.5)[0]
    assert len(stored("measurements")) == 1


def test_production_availability_performance_quality_message():
    # availability 400/480 = 83.3%, performance 360/(1*400) = 90.0%, quality 342/360 = 95.0%
    ok, msg = store.add_production("2026-01-01 06:00", "2026-01-01 14:00", "Line 1", "P1",
                                   480, 400, 1.0, 360, 342)
    assert ok
    assert "availability 83.3%" in msg and "performance 90.0%" in msg and "quality 95.0%" in msg
    assert stored("production").iloc[0]["shift"] == "Day"


def test_production_validation():
    args = ("2026-01-01 06:00", "2026-01-01 14:00", "Line 1", "P1")
    assert not store.add_production(*args, 480, 500, 1.0, 360, 342)[0]      # run > planned
    assert not store.add_production(*args, 480, 400, 1.0, 360, 400)[0]      # good > total
    assert not store.add_production(*args, 480, 400, 0, 360, 342)[0]        # ideal rate zero
    assert not store.add_production("2026-01-01 14:00", "2026-01-01 06:00",
                                    "Line 1", "P1", 480, 400, 1.0, 360, 342)[0]  # end before start
    assert len(stored("production")) == 0


def test_downtime_minutes_come_from_the_times():
    ok, msg = store.add_downtime("2026-01-01 10:00", "2026-01-01 10:45", "Line 1", "Changeover")
    assert ok and "45 minutes" in msg
    assert stored("downtime").iloc[0]["minutes"] == 45
    assert not store.add_downtime("2026-01-01 10:45", "2026-01-01 10:00", "Line 1", "Changeover")[0]


def test_defect_validation():
    assert store.add_defect("2026-01-01 10:00", "Line 1", "P1", "Scratches", 12.5, lot="L1")[0]
    assert not store.add_defect("2026-01-01 10:10", "Line 1", "P1", "Scratches", 0)[0]
    assert not store.add_defect("2026-01-01 10:00", "Line 1", "P1", "Scratches", 5)[0]  # duplicate
    assert len(stored("defects")) == 1


def test_spec_validation_and_update():
    assert not store.add_spec("P1", "od", 11.0, 10.0)[0]
    assert store.add_spec("P1", "wall", 9.9, 11.2, 10.5, "mm")[0]          # replaces the old limits
    assert store.get_spec("P1", "wall")["lsl"] == 9.9


def test_import_dataframe_reports_good_and_bad_rows():
    raw = pd.DataFrame({"Timestamp": ["2026-01-01 08:00", "2026-01-01 08:30", "bad"],
                        "Line": ["Line 1"] * 3, "Product": ["P1"] * 3,
                        "Characteristic": ["wall"] * 3, "Value": [10.4, "x", 10.6]})
    raw.columns = [c.lower() for c in raw.columns]
    r = importer.import_dataframe(raw, "measurements")
    assert r["imported"] == 1 and r["rejected"] == 2
    assert len(stored("measurements")) == 1
