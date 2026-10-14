"""Plain-language report from the computed facts, with an automatic number check.

The language model only writes sentences. Every number and date in its text is checked
against the facts; a report with a number that is not in the facts is NOT marked verified.
"""
import json
import re
import time

MODEL = "qwen2.5:7b"          # the one setting to change for another model
MAX_ATTEMPTS = 2              # first try plus one retry with feedback
OPTIONS = {"temperature": 0, "seed": 0}

NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
TIME = re.compile(r"\b\d{1,2}:\d{2}\b")

RULES = """You are a quality assistant at a polymer pipe extrusion plant. Write a short, \
plain-English report from the FACTS the user sends.

Rules:
1. Use only numbers that appear in the FACTS, copied exactly as written (for example 75.1, \
not 75). Never calculate a new number: no sums, differences, averages, ratios or changes.
2. Write every number with digits, never with words.
3. Write dates and times exactly as in the FACTS (YYYY-MM-DD HH:MM).
4. Do not state causes and do not give advice that the FACTS do not support. You may name \
what looks worth checking, based only on the FACTS (the biggest loss, the top defect, a note).
5. Mention every item under notes in plain words. Name the line and the period (from and to, \
copied exactly) in your first sentence. Write SPC and AI in capital letters.
6. cpk is short-term capability and ppk is overall capability of the wall thickness. Use the \
word in cpk_rating instead of quoting a target value.
7. spc_alarm_events are groups of control-chart alarms. ai_anomaly_events are groups of \
readings the AI flagged as unusual.
8. If a section is null, say that no data is available for it.
9. If percent_out_of_spec is above 0, say so next to the capability.

Format: one short paragraph (3 to 5 sentences) about this one line, covering capability, alarms \
and anomalies, OEE with its biggest loss, and scrap with the top defects. Then one line starting \
with "Worth checking:" with at most two items. No headings, no lists, no markdown."""


class LLMUnavailable(Exception):
    """Ollama could not be reached or the model gave no answer."""


def build_messages(facts_list):
    """The system rules plus the facts of one line as compact JSON."""
    facts = json.dumps(facts_list, separators=(",", ":"), default=str)
    return [{"role": "system", "content": RULES},
            {"role": "user", "content": "FACTS:\n" + facts}]


def _norm_time(text):
    hour, minute = text.split(":")
    return f"{int(hour)}:{minute}"


def _split_dates(text):
    """The text without dates and times, plus the dates and times found in it."""
    dates = set(DATE.findall(text))
    rest = DATE.sub(" ", text)
    times = {_norm_time(t) for t in TIME.findall(rest)}
    return TIME.sub(" ", rest), dates, times


def _decimals(text):
    return len(text.split(".")[1]) if "." in text else 0


def _as_float(text):
    return float(text.replace(",", ""))


def _collect(node, numbers, dates, times):
    """Walk the facts and collect every number (value, decimals) and every date and time."""
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, (int, float)):
        text = format(float(node), ".10g")
        numbers.append((float(node), _decimals(text)))
    elif isinstance(node, str):
        rest, d, t = _split_dates(node)
        dates |= d
        times |= t
        numbers.extend((_as_float(n), _decimals(n)) for n in NUMBER.findall(rest))
    elif isinstance(node, dict):
        for value in node.values():
            _collect(value, numbers, dates, times)
    else:
        for value in node:
            _collect(value, numbers, dates, times)


def _matches(token, numbers):
    """A reported number matches a fact when it is that fact or the fact rounded, but a
    fact with decimals keeps at least one decimal (0.897 may be 0.9 but not 1)."""
    x, d = _as_float(token), _decimals(token)
    for value, dv in numbers:
        if d > dv:
            if abs(x - value) < 1e-9:
                return True
        elif d >= min(dv, 1) and abs(x - value) <= 0.5 * 10 ** (-d) + 1e-9:
            return True
    return False


def check_numbers(text, facts_list):
    """Numbers, dates and times in the text that are NOT in the facts (empty = verified)."""
    numbers, dates, times = [], set(), set()
    _collect(facts_list, numbers, dates, times)
    rest, found_dates, found_times = _split_dates(text)
    bad = [d for d in sorted(found_dates) if d not in dates]
    bad += [t for t in sorted(found_times) if t not in times]
    for token in NUMBER.findall(rest):
        if not _matches(token, numbers) and token not in bad:
            bad.append(token)
    return bad


def ask_ollama(model, messages):
    """The real model call (the tests replace this with a fake)."""
    try:
        import ollama
        reply = ollama.chat(model=model, messages=messages, options=OPTIONS)
        return reply["message"]["content"].strip()
    except Exception as error:
        raise LLMUnavailable(f"Could not get an answer from Ollama ({error}). Is Ollama "
                             f"running, and is the model '{model}' installed?") from error


def _write_one(facts, chat, model):
    """One line's paragraph, checked against that line's facts only. Retries once."""
    messages = build_messages(facts)
    text, bad = "", []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        text = chat(model, messages)
        bad = check_numbers(text, facts)
        if not bad:
            break
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "This text contains numbers that are not in the FACTS: "
             + ", ".join(bad) + ". Rewrite the whole report. Use only numbers from the FACTS, "
             "copied exactly, and remove every other number."}]
    return text, bad, attempt


def write_report(facts_list, chat=ask_ollama, model=MODEL):
    """Write one paragraph per line and verify its numbers against that line's facts only,
    so a number can never be taken from the other line. facts_list has one entry per line."""
    if not facts_list:
        raise ValueError("No lines to write a report about")
    start = time.time()
    texts, lines, bad_all, attempts = [], [], [], 1
    for facts in facts_list:
        text, bad, used = _write_one(facts, chat, model)
        texts.append(text)
        lines.append({"line": facts.get("line"), "ok": not bad, "bad_numbers": bad,
                      "attempts": used})
        bad_all += [n for n in bad if n not in bad_all]
        attempts = max(attempts, used)
    return {"text": "\n\n".join(texts), "ok": not bad_all, "bad_numbers": bad_all,
            "attempts": attempts, "lines": lines, "model": model,
            "seconds": round(time.time() - start, 1)}