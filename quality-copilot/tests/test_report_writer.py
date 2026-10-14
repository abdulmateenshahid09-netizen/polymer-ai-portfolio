"""Tests for the report writer. A fake model replaces Ollama, so no model is needed."""
import json

import pandas as pd
import pytest

import simulator
from ai.report_facts import build_facts
from ai.report_writer import (LLMUnavailable, MAX_ATTEMPTS, ask_ollama, build_messages,
                              check_numbers, write_report)

FACTS = [{
    "line": "Line 1", "period": "Last 7 days", "from": "2026-10-07 05:30", "to": "2026-10-14 06:00",
    "wall_thickness": {"product": "PE100 SDR11 110mm", "readings": 336, "cpk": 1.831,
                       "ppk": 0.897, "cpk_rating": "excellent", "spc_alarm_events": 7,
                       "ai_anomaly_events": 4},
    "production": {"oee_percent": 70.6, "scrap_percent": 5.0, "planned_minutes": 10080.0,
                   "top_defects": [{"type": "Surface scratches", "percent_of_scrap": 31.2}]},
    "notes": ["Only 25 readings (fewer than 30): treat the indices as indicative only."],
}]

GOOD = ("Line 1, 2026-10-07 05:30 to 2026-10-14 06:00: Cpk 1.831 and Ppk 0.9 over 336 readings. "
        "There were 7 alarm events and 4 AI events. OEE was 70.6% and scrap 5%. "
        "Surface scratches caused 31.2% of scrap. Planned time was 10,080 minutes.")


def test_good_text_passes():
    assert check_numbers(GOOD, FACTS) == []


@pytest.mark.parametrize("text, bad", [
    ("OEE was 80.2%.", ["80.2"]),                       # invented number
    ("OEE was 70.6% and scrap 5%, a gap of 65.6.", ["65.6"]),   # a calculated number
    ("There were 8 alarm events.", ["8"]),
    ("OEE was 71%.", ["71"]),                           # 70.6 keeps one decimal
    ("The period began 2026-10-08 05:30.", ["2026-10-08"]),   # wrong date
    ("The period ended at 07:45.", ["7:45"]),           # wrong time
])
def test_wrong_numbers_are_found(text, bad):
    assert check_numbers(text, FACTS) == bad


def test_a_fact_with_decimals_cannot_be_rounded_to_a_whole_number():
    # "Line 1" would make a plain 1 legitimate, so use a line name without a digit
    facts = json.loads(json.dumps(FACTS).replace("Line 1", "Line A"))
    assert check_numbers("Ppk was 1.", facts) == ["1"]       # 0.897 may be 0.9, not 1
    assert check_numbers("Ppk was 0.9.", facts) == []


def test_rounding_rules():
    assert check_numbers("Ppk 0.90 and Cpk 1.83.", FACTS) == []
    assert check_numbers("Scrap 5.0 percent.", FACTS) == []
    assert check_numbers("Planned 10080 minutes.", FACTS) == []
    assert check_numbers("Cpk 1.7.", FACTS) == ["1.7"]


def test_every_number_of_real_facts_is_allowed():
    meas, _p, prod, down, defects, _e = simulator.build_tables()
    specs = pd.DataFrame({"product": ["PE100 SDR11 110mm"], "characteristic": ["wall_thickness"],
                          "lsl": [10.0], "usl": [11.1], "target": [10.4]})
    meas["timestamp"] = pd.to_datetime(meas["timestamp"])
    for df in (prod, down):
        df["start_time"] = pd.to_datetime(df["start_time"])
    defects["timestamp"] = pd.to_datetime(defects["timestamp"])
    end = max(meas["timestamp"].max(), pd.to_datetime(prod["end_time"]).max())
    facts = [build_facts(line, "Last 7 days", end - pd.Timedelta(days=7), end, meas, specs,
                         prod, down, defects) for line in ("Line 1", "Line 2")]
    values = []

    def walk(node):
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            values.append(str(node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(facts)
    assert len(values) > 30
    assert check_numbers(" ; ".join(values), facts) == []
    assert check_numbers("OEE was 12.34 percent.", facts) == ["12.34"]


def test_prompt_contains_facts_and_rules():
    system, user = build_messages(FACTS)
    assert system["role"] == "system" and "copied exactly" in system["content"]
    assert json.loads(user["content"].split("FACTS:\n")[1]) == FACTS


def fake(*answers):
    """A fake model that gives the answers in turn and remembers what it was asked."""
    calls = []

    def chat(model, messages):
        calls.append(messages)
        return answers[min(len(calls), len(answers)) - 1]
    chat.calls = calls
    return chat


def test_verified_report_needs_one_attempt():
    chat = fake(GOOD)
    r = write_report(FACTS, chat=chat, model="m")
    assert r["ok"] and r["attempts"] == 1 and r["text"] == GOOD and r["bad_numbers"] == []
    assert r["model"] == "m"


def test_retry_names_the_bad_numbers_and_can_fix_the_report():
    chat = fake("OEE was 80.2%.", GOOD)
    r = write_report(FACTS, chat=chat)
    assert r["ok"] and r["attempts"] == 2 and r["text"] == GOOD
    retry = chat.calls[1]
    assert retry[-2] == {"role": "assistant", "content": "OEE was 80.2%."}
    assert "80.2" in retry[-1]["content"]


def test_report_with_bad_numbers_is_never_marked_verified():
    chat = fake("OEE was 80.2%.", "OEE was 81.5%.")
    r = write_report(FACTS, chat=chat)
    assert not r["ok"] and r["attempts"] == MAX_ATTEMPTS and r["bad_numbers"] == ["81.5"]


def test_missing_ollama_gives_a_friendly_error(monkeypatch):
    import ollama

    def broken(**kwargs):
        raise ConnectionError("refused")
    monkeypatch.setattr(ollama, "chat", broken)
    with pytest.raises(LLMUnavailable, match="Is Ollama running"):
        ask_ollama("m", [])