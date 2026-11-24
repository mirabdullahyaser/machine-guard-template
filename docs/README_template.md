# Machine Guard

<!--
Capstone README template (Day 4). Copy this file over the repo's README.md at the capstone:
    cp docs/README_template.md README.md
then fill in every TODO. Most of it is already written: the capstone tests what you understand,
not how fast you write. Delete this comment when you are done.
-->

Predicts which milling-machine runs are about to fail, so maintenance can step in before a breakdown. Built in four days, from a notebook to a monitored, automatically tested service.

**Live API:** `TODO` · **Streamlit page:** `TODO` · **CI:** `TODO: status badge, see below`

<!-- Status badge: replace <you> with your GitHub user name -->
<!-- ![CI](https://github.com/<you>/machine-guard/actions/workflows/ci.yml/badge.svg) -->

## The business problem

Unplanned downtime is expensive. About 3.4% of runs end in a failure. A model that always says "no failure" is 96.6% accurate and useless. We judge the model on **recall** (breakdowns caught) and count **false alarms** (wasted technician visits), at a cost ratio of about 20 to 1.

`TODO: one sentence in your own words on what a missed breakdown costs on a real shop floor.`

## Results

| | Value |
|---|---|
| Champion model | `TODO: model type, MLflow version` |
| Recall on failures | `TODO` (gate: at least 0.75) |
| Precision | `TODO` |
| Missed breakdowns / false alarms on the 2,000-run test set | `TODO` / `TODO` |

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the diagram and the design choices.

## How to run it

```bash
pip install -r requirements.txt
python -m machine_guard.validate data/raw/ai4i2020.csv --require-target   # check the data
pytest                                                                    # unit tests
python -m machine_guard.quality_gate                                      # is the model good enough?
uvicorn app.main:app --port 8000                                          # serve (set API_KEY first)
python -m machine_guard.drift --current data/incoming/next_quarter.csv    # drift report in reports/
python flows/retrain_flow.py                                              # retrain, promote if it passes
```

## Operating it

- **Monitoring:** every prediction is logged to `logs/predictions.db`. A weekly GitHub Action checks drift and opens an issue.
- **Incidents and rollback:** [`docs/runbook.md`](docs/runbook.md).
- **Model facts, limits and approval:** [`docs/model_card.md`](docs/model_card.md).

## Maturity

Google MLOps maturity: **small Level 1**. Automated tests, a quality gate and a retraining pipeline exist. `TODO: one thing that is still manual, and what you would automate next.`

## Credits

Data: Stephan Matzka, "AI4I 2020 Predictive Maintenance Dataset", UCI Machine Learning Repository, 2020, CC BY 4.0. https://doi.org/10.24432/C5HS5C
