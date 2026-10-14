"""Tests for the simulated live stream. Each test uses its own temporary database."""
import numpy as np
import pandas as pd
import pytest

import db
import live
import store


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "live.db")
    db.init_db()
    store.add_spec("P1", "wall", 10.0, 11.1, 10.4, "mm")
    rng = np.random.default_rng(1)
    base = pd.Timestamp("2026-01-01 06:00")
    values = 10.45 + np.cumsum(rng.normal(0, 0.02, 60)) * 0.2 + rng.normal(0, 0.05, 60)
    for i, v in enumerate(values):
        store.add_measurement(base + pd.Timedelta(minutes=30 * i), "Line 1", "P1", "wall",
                              round(float(v), 3), lot="L1")
    return db.DB_PATH


def history(seed=3, n=200):
    rng = np.random.default_rng(seed)
    x, out = 0.0, []
    for _ in range(n):
        x = 0.7 * x + rng.normal(0, 0.05)
        out.append(10.45 + x)
    return out


def test_next_value_stays_close_to_the_history():
    rng = np.random.default_rng(0)
    h = history()
    draws = np.array([live.next_value(h, rng) for _ in range(3000)])
    # one step ahead of the same history: centred near mean + r * (last - mean)
    expected = np.mean(h) + 0.7 * (h[-1] - np.mean(h))
    assert draws.mean() == pytest.approx(expected, abs=0.02)
    assert 0.02 < draws.std() < 0.12


def test_fault_offset_shifts_the_readings():
    h = history()
    a = np.mean([live.next_value(h, np.random.default_rng(i)) for i in range(2000)])
    b = np.mean([live.next_value(h, np.random.default_rng(i), offset=0.3) for i in range(2000)])
    assert b - a == pytest.approx(0.3, abs=1e-9)   # same random draws, so exactly the offset


def test_needs_enough_history():
    with pytest.raises(ValueError):
        live.next_value([10.4, 10.5, 10.4], np.random.default_rng(0))


def test_append_adds_a_labelled_reading_one_step_later(database):
    before = db.read_table("measurements")
    last = pd.Timestamp(before["timestamp"].max())
    ok, _, when, value = live.append_simulated("Line 1", "P1", "wall", np.random.default_rng(0))
    assert ok and when == last + pd.Timedelta(minutes=30)
    after = db.read_table("measurements")
    new = after[after["notes"] == live.NOTE]
    assert len(after) == len(before) + 1 and len(new) == 1
    assert new.iloc[0]["operator"] == "Simulated" and new.iloc[0]["lot"] == "L1"
    assert new.iloc[0]["value"] == pytest.approx(value)


def test_delete_removes_only_simulated_readings(database):
    n_real = len(db.read_table("measurements"))
    rng = np.random.default_rng(0)
    for _ in range(5):
        live.append_simulated("Line 1", "P1", "wall", rng)
    assert live.count_simulated() == 5
    assert live.delete_simulated() == 5
    assert live.count_simulated() == 0
    assert len(db.read_table("measurements")) == n_real


def test_append_refuses_when_there_is_too_little_history(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "empty.db")
    ok, message, _, _ = live.append_simulated("Line 9", "P1", "wall", np.random.default_rng(0))
    assert not ok and "at least 10" in message
