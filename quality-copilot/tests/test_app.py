"""App tests: every page runs, entries can be saved, and saved entries show up on the
charts. They run on a temporary database filled with demo data, never on your own data."""
import base64
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import db
import simulator
import store

PAGES = ["overview", "entry", "spc", "oee", "defects", "live_monitor", "ai_insights"]


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("app") / "demo.db"
    patch = pytest.MonkeyPatch()
    patch.setattr(db, "DB_PATH", path)
    patch.setattr(simulator, "DB_PATH", path)
    db.init_db()
    store.add_spec("PE100 SDR11 110mm", "wall_thickness", 10.0, 11.1, 10.4, "mm")
    store.add_spec("PE100 SDR11 110mm", "outside_diameter", 110.0, 110.3, 110.15, "mm")
    simulator.load_demo_data()
    yield path
    patch.undo()


def run_page(page):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return AppTest.from_file(str(root / "views" / f"{page}.py"), default_timeout=90).run()


def run_app():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return AppTest.from_file(str(root / "app.py"), default_timeout=90).run()


def decode(values):
    if isinstance(values, dict) and "bdata" in values:
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=np.dtype(values["dtype"]))
    return np.asarray(values)


def figure_traces(at, index=0):
    spec = json.loads(at.get("plotly_chart")[index].proto.spec)
    return {t.get("name"): t for t in spec["data"]}


@pytest.mark.parametrize("page", PAGES)
def test_every_page_runs_without_errors(demo_db, page):
    at = run_page(page)
    assert not at.exception


def test_whole_app_starts_and_has_the_filters(demo_db):
    at = run_app()
    assert not at.exception
    assert [s.key for s in at.sidebar.selectbox] == ["flt_line", "flt_period"]


def test_line_filter_changes_the_spc_page(demo_db):
    at = run_app()
    at.switch_page("views/spc.py").run()
    assert len(at.tabs) == 2                      # one tab per line
    at.sidebar.selectbox(key="flt_line").select("Line 2").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1       # just the chosen line


def test_key_numbers_are_shown(demo_db):
    metrics = {m.label for m in run_page("spc").metric}
    assert {"Cpk (short-term)", "Ppk (overall)", "Cp", "Pp"} <= metrics
    oee = {m.label: m.value for m in run_page("oee").metric}
    assert set(oee) == {"Availability", "Performance", "Quality", "OEE"}
    assert oee["OEE"].endswith("%")


def test_saving_a_reading_updates_the_chart(demo_db):
    when = pd.Timestamp.now().floor("min") + pd.Timedelta(days=1)
    ok, _ = store.add_measurement(when, "Line 2", "PE100 SDR11 110mm", "wall_thickness", 11.6)
    assert ok
    at = run_app()
    at.sidebar.selectbox(key="flt_line").select("Line 2").run()
    at.switch_page("views/spc.py").run()
    traces = figure_traces(at)
    assert np.any(np.isclose(decode(traces["Reading"]["y"]), 11.6))
    assert np.any(np.isclose(decode(traces["Alarm"]["y"]), 11.6))   # marked as an alarm


def save_button(at, label):
    return [b for b in at.button if b.label == label][0]


def test_entry_form_refuses_a_typing_mistake_then_saves_a_real_value(demo_db):
    at = run_page("entry")
    assert at.selectbox(key="m_char").value == "wall_thickness"      # remembers the last choice
    before = len(db.read_table("measurements"))
    at.number_input[0].set_value(104.5)
    save_button(at, "Save reading").click()
    at.run()
    assert at.error and "typing mistake" in at.error[0].value
    assert len(db.read_table("measurements")) == before

    at.number_input[0].set_value(10.55)
    save_button(at, "Save reading").click()
    at.run()
    assert at.success and "within specification" in at.success[0].value
    assert len(db.read_table("measurements")) == before + 1


def test_production_form_checks_the_numbers(demo_db):
    at = run_page("entry")
    boxes = {n.label: n for n in at.number_input}
    boxes["Planned minutes"].set_value(480.0)
    boxes["Run minutes (planned minus downtime)"].set_value(500.0)
    save_button(at, "Save production run").click()
    at.run()
    assert at.error and "Run minutes" in at.error[0].value


def test_empty_database_shows_friendly_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "empty.db")
    for page in PAGES:
        at = run_page(page)
        assert not at.exception, page


def test_live_page_streams_readings_and_shows_a_fault_as_an_alarm(demo_db):
    import live
    at = run_page("live_monitor")
    assert not at.exception
    before = len(db.read_table("measurements"))
    at.toggle(key="live_on").set_value(True).run()
    at.toggle(key="live_stream").set_value(True).run()
    assert not at.exception
    assert live.count_simulated() >= 2            # one new reading per line on each refresh
    assert len(db.read_table("measurements")) > before

    [b for b in at.button if b.label.startswith("Inject a fault")][0].click()
    at.run()
    seen_alarm = False
    for _ in range(6):
        at.run()
        seen_alarm = seen_alarm or any("ALARM" in e.value for e in at.error)
    assert seen_alarm, "a fault of about 3 sigma should trigger an alarm within a few readings"

    [b for b in at.button if b.label == "Remove simulated readings"][0].click()
    at.run()
    assert live.count_simulated() == 0                 # removed, and not re-added
    assert at.toggle(key="live_stream").value is False  # the stream was switched off
    assert at.success and "Removed" in at.success[0].value


def test_ai_insights_shows_anomalies_for_each_line(demo_db):
    at = run_app()
    at.switch_page("views/ai_insights.py").run()
    assert not at.exception
    assert len(at.tabs) == 2                              # one tab per line
    assert len(at.get("plotly_chart")) == 2
    labels = {m.label for m in at.metric}
    assert {"SPC alarm events", "AI anomaly events"} <= labels
    at.sidebar.selectbox(key="flt_line").select("Line 1").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1
