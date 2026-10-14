# Quality Analytics Copilot

A dashboard for a production quality engineer (PE pipe extrusion). You type in or import production records, and it instantly gives capability (Cp, Cpk, Pp, Ppk), SPC control charts with alarm rules, OEE with a loss breakdown and defect analysis. An AI layer flags anomalies, scores off-target risk and suggests haul-off speed changes, and a local language model writes a plain-language report from the computed numbers.

**Honest framing:** all data in this repository is simulated (30 days for two lines, seed 42). "Real time" means the dashboard updates the moment an entry is saved or a CSV/Excel file is imported. It is not connected to machines or PLCs. That would need plant IT approval and a gateway such as OPC UA.

**Design rule:** Python does every calculation. The language model only writes sentences from numbers it is given, and every number, date and time in its text is checked automatically against those numbers.

## What is in it

The dashboard has 8 pages:

| Page | What it shows |
| --- | --- |
| Overview | Key numbers for the chosen line and period |
| Data entry | Validated forms for readings, production runs, downtime, defects and specs, plus CSV/Excel import |
| SPC and capability | Control charts, Western Electric rules, Cp/Cpk/Pp/Ppk |
| Live monitor | A simulated stream of readings |
| OEE | Availability, performance, quality, loss breakdown |
| Defects | Pareto chart, scrap by defect and by lot |
| AI insights | Anomalies next to SPC alarms, off-target risk model against simple rules, settings recommender |
| Shift report | A local model writes one paragraph per line, with an automatic number check |

## How to run it

You need Python, Git and (for the report page only) [Ollama](https://ollama.com).

```
git clone https://github.com/abdulmateenshahid09-netizen/polymer-ai-portfolio.git
cd polymer-ai-portfolio
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r quality-copilot\requirements.txt
python quality-copilot\add_specs.py
python quality-copilot\simulator.py
streamlit run quality-copilot\app.py
```

For the Shift report page, install the model once with `ollama pull qwen2.5:7b` and keep Ollama running. Writing the report for two lines takes about 2 minutes on a laptop with 16 GB RAM.

Run the tests with `python -m pytest quality-copilot\tests -q` (137 tests).

## Results, including the ones that are not impressive

**Anomaly detection (Isolation Forest)** against SPC rules 1-2, on the simulated faults:

| Line | Method | Faults caught | False alarms per day |
| --- | --- | --- | --- |
| Line 1 | SPC rules 1-2 | 2 of 2 | 0.51 |
| Line 1 | Isolation Forest | 2 of 2 | 0.43 |
| Line 2 | SPC rules 1-2 | 1 of 1 | 0.62 |
| Line 2 | Isolation Forest | 1 of 1 | 0.31 |

Same speed as the SPC rules, modestly fewer false alarms.

**Off-target risk (XGBoost)** against simple rules, time-based split (train on the first 20 days, test on the last 10):

| Method | AUC | Average precision |
| --- | --- | --- |
| XGBoost model | 0.647 | 0.237 |
| Naive abs(z) rule | 0.648 | 0.327 |

**The model does not beat the simple rule.** Only 26 of 2880 readings are actually out of spec, the simulated faults start abruptly so there is no early warning to learn, and 83 % of the readings the model rates HIGH are night shift (33 % of all test readings), so a high score often just means "it is night". The dashboard shows this verdict openly.

**Settings recommender:** a plain regression of wall thickness on the settings explained almost nothing, because operators change haul speed in response to the readings. The recommender is therefore physics first: a mass balance back-calculates extruder output, a small model learns how output depends on melt temperature and MFR, and Optuna searches the haul-off speed within a limit. On the last 10 days it predicts wall thickness with an error of 0.069 mm, against 0.094 mm for "next reading = last reading".

## Limitations

- All data is simulated. Nothing here proves the tools work on a real line.
- The anomaly detector is about as good as the SPC rules, and the risk model is no better than a simple rule.
- The recommender is validated only as a thickness prediction on held-out simulated data. The effect of following its advice was not tested on a real line.
- The Shift report is text written by a 7B model. The number check guards numbers, dates and times only, not wording, and the text can still be incomplete or awkward.
- It is a single-user prototype, with no login and no connection to machines.

## Using it on a real line

Import real measurements with the CSV/Excel importer, enter your own specs, and treat the AI pages as experiments until they have been tested on your own data. Get approval before using production data, and do not publish it.

## Project layout

- `app.py`, `ui.py`, `charts.py`, `views/`: the dashboard
- `metrics/`: capability, SPC, OEE and defect calculations
- `ai/`: anomaly detection, risk model, recommender, report facts and report writer
- `db.py`, `store.py`, `importer.py`, `simulator.py`, `live.py`: data storage, entry, import and simulation
- `tests/`: 137 tests, including deliberate-breakage checks