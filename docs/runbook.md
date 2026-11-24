# Machine Guard incident runbook

What to do when something goes wrong with Machine Guard, step by step. Written so that someone who did not build the system can follow it at 3 a.m.

Fill in every `TODO` (Day 4, Lab 2). Keep the commands exact: copy them from a run that worked.

## 1. Service facts

| Item | Value |
|---|---|
| What it does | Predicts whether a milling-machine run will fail, from six sensor readings |
| Live API | `TODO: https://....onrender.com` |
| Streamlit page | `TODO: https://....streamlit.app` |
| Image | `ghcr.io/TODO-github-user/machine-guard:<tag>` |
| Model in production | `TODO: version N` (check: `curl <API>/health` shows `model_version`) |
| Previous good model | `TODO: version N-1` (the rollback target) |
| Quality gate | Recall on failures at least **0.75** on the course test set |
| Owner (first call) | `TODO: name, phone or chat handle` |
| Backup | `TODO: name` |

## 2. How bad is it?

| Severity | Examples | Respond within | Who |
|---|---|---|---|
| **SEV1** | API down; every prediction is "no failure"; a wrong model is live | 30 minutes | Owner, then backup |
| **SEV2** | Drift alert on key sensors; CI red on `main`; latency several times normal | 1 working day | Owner |
| **SEV3** | One odd prediction; drift on a minor column; a flaky test | This week | Anyone |

Rule of thumb: if machines could break down without warning because of us, it is SEV1.

## 3. Playbooks

### 3.1 Drift alert (a GitHub issue labelled `drift` was opened)

1. Open the issue. Note **which columns** drifted and their scores.
2. Download the HTML report from the workflow run (artifact `drift-report`) and look at the drifted columns' distributions. Is it a shift (all values moved), a stretch, or a new cluster?
3. Ask the plant: did anything change? New supplier, sensor recalibration, new product type, a new line? `TODO: who to ask`.
4. Check data validation: `python -m machine_guard.validate <batch.csv>`. Impossible values are a data bug, not drift: fix the feed, don't retrain.
5. If labels for the new period exist, measure the champion on them (`python -m machine_guard.quality_gate --batch <labelled batch.csv>`, which grades on every row of the batch). Recall below 0.75 means the model is now missing breakdowns: treat it as **SEV1**.
6. Decide and write the decision in the issue:
   - **Real change in the world, model still fine:** close the issue with a note.
   - **Real change, model worse:** run the retraining flow (`python flows/retrain_flow.py --batch <batch.csv>`), then ship the new champion (3.3).
   - **Broken sensor or feed:** fix upstream; don't retrain on broken data.

### 3.2 Bad model in production: roll back to the previous registered model

Symptoms: recall drop reported by the shop floor, many more or fewer alarms than usual, CI quality gate red, `/health` shows an unexpected `model_version`.

**Fast path (minutes): undo the commit that changed the model files.** The model ships as committed files, so Git can undo it.

```bash
git log --oneline -- models/            # find the commit that brought the bad model
git revert <commit>                     # makes a new commit that restores the previous files
git push                                # CI tests, gates and builds a new image
```

**Registry path: move the alias back, then export.**

```bash
# 1. Point @champion back at the previous good version (TODO: fill in the number)
python -c "from mlflow import MlflowClient; import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); MlflowClient().set_registered_model_alias('machine-guard', 'champion', TODO)"

# 2. Write models/model.joblib and models/model_meta.json from @champion
python -m machine_guard.export_model

# 3. Check it before shipping
python -m machine_guard.quality_gate

# 4. Ship
git add models/
git commit -m "Roll back to model version TODO: <reason>"
git push
```

**Redeploy** once CI is green (the image is tagged with the commit SHA):

- Render: `TODO: how your service picks up a new image (auto-deploy or Manual Deploy > Deploy latest)`.
- kind: `kubectl set image deployment/machine-guard api=ghcr.io/TODO-github-user/machine-guard:<sha>`, or `kubectl rollout undo deployment/machine-guard` to return to the previous image straight away.

**Verify:** `curl <API>/health` shows the previous `model_version`; send one known run to `/predict` and check the answer is sensible.

### 3.3 Ship a new champion (after retraining)

1. The retraining flow moved `@champion` only if the gate passed. Check in the MLflow UI.
2. `python -m machine_guard.export_model`, then `git add models/ && git commit -m "Model version N: <why>" && git push`.
3. Wait for CI to go green; redeploy as in 3.2.
4. Update the model card (`docs/model_card.md`) and section 1 of this runbook.

### 3.4 API down or erroring

1. `curl <API>/health`. Render's free tier sleeps after 15 idle minutes: the first call can take about a minute.
2. Render dashboard > Logs (or `kubectl logs deployment/machine-guard`). Look for the last error line.
3. Every call returns 401: has `API_KEY` changed in Render's environment settings? Every call returns 422: the caller is sending invalid input (Celsius?).
4. Still broken: roll back to the previous image (3.2, redeploy step).

### 3.5 CI is red on `main`

1. Open the failed run in the Actions tab and read the **first** red step.
2. Validation failed: the data changed. Tests failed: the code changed. Gate failed: the model got worse. Image failed: Dockerfile or registry permissions.
3. Never "fix" a red gate by lowering the gate in the code. Fix the cause or roll back.

## 4. After every SEV1 or SEV2: a short review

Blameless: we look for what in the system let this happen, not who.

```text
Date and duration:
What users saw:
Timeline (detected, acknowledged, mitigated, resolved):
Root cause:
What caught it (or should have):
What we change so it can't happen again (owner, date):
```
