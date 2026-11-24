# Model card: Machine Guard failure predictor

A model card is a one-page fact sheet for a model: what it is for, how good it is, where it fails, and who approved it. Fill in every `TODO` (Day 4, governance block). Keep it short and honest.

## Model details

| Item | Value |
|---|---|
| Name | `machine-guard` (MLflow registered model) |
| Version in production | `TODO` (alias `champion`) |
| Type | Random forest classifier, `class_weight="balanced"`, inside a scikit-learn pipeline with two engineered features (`power_w`, `temp_diff_k`) |
| Inputs | `type` (L/M/H), `air_temp_k`, `process_temp_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min` |
| Output | Failure probability; "failure" when the probability is at least 0.5 |
| Training data | AI4I 2020 Predictive Maintenance dataset (UCI, CC BY 4.0), 10,000 runs, 3.39% failures. Data version (md5): `TODO` |
| Code version | Git commit `TODO` |
| Owner | `TODO` |

## Intended use

- **For:** `TODO: one sentence. Who uses the prediction, and what do they do with it?`
- **Not for:** `TODO: one thing it must not be used for` (for example, judging operators' performance, or machines and sensors it was not trained on).

## Performance (course test set: 2,000 runs, 68 failures)

| Metric | Value |
|---|---|
| Recall on failures (breakdowns caught) | `TODO` |
| Precision (alarms that were real) | `TODO` |
| Missed breakdowns / false alarms | `TODO` / `TODO` |
| Quality gate | Recall at least 0.75: `TODO pass/fail` |

## Fairness and slices (from your Day 2 slice note)

`TODO: missed failures by product type L, M and H. Is any type served worse? What did you decide to do about it?`

## Limitations and risks

- `TODO: what kind of failure does it miss most?`
- Drift: a sensor change or a new tool supplier changes the inputs. The weekly drift alert watches for this (`docs/runbook.md`, 3.1).
- The data is synthetic (a published benchmark), not from your plant. Re-validate on real data before relying on it.
- `TODO: anything else`

## Explainability

`TODO: which readings push predictions toward failure most (from your Day 2 SHAP plot)?`

## Privacy and security

- No personal data: inputs are machine sensor readings only.
- The API needs a key (`X-API-Key`); the key lives in environment variables and secrets, never in Git.

## Approval

| Step | Who | Date | Decision |
|---|---|---|---|
| Quality gate in CI | automatic | `TODO` | `TODO pass/fail` |
| Technical review | `TODO` | | |
| Business sign-off (maintenance lead) | `TODO` | | |
