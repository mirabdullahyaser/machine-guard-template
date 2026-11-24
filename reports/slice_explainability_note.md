# Machine Guard: slice and explainability note

One page for a maintenance manager who must decide whether to trust the model on all three product lines.
Plain words. Every number from `notebooks/05_slices_fairness_shap.ipynb`, with its count next to it.

**Model:** `machine-guard` version ___, alias `champion` (MLflow run `______`)
**Data:** md5 `________` (DVC tag `data-v__`), ______ runs, test set ______ runs with ___ failures
**Date and author:** ______

## 1. Overall

Out of ___ real failures in the test period, the model caught ___ (recall ___) and raised ___ false alarms (precision ___).

## 2. Slices: product types L, M and H

| Type | Test runs | Failures | Recall | Missed | False alarms |
|---|---|---|---|---|---|
| L | | | | | |
| M | | | | | |
| H | | | | | |

Across all 10,000 runs (cross-validated), recall by type: L ___ (___ failures), M ___ (___), H ___ (___).

**What this means:** which product type does the model serve worst? Is the evidence strong (many failures) or weak (a handful)?

> ...

## 3. Which failures does it miss?

| Failure mode | Caught | Missed |
|---|---|---|
| Tool wear (TWF) | | |
| Heat dissipation (HDF) | | |
| Power (PWF) | | |
| Overstrain (OSF) | | |
| Random (RNF) | | |

**What this means on the shop floor:**

> ...

## 4. Explanations (SHAP)

- **Readings that matter most overall:** ...
- **One alarm, explained:** run ___, probability ___. The readings that pushed it towards "fail": ...
- **One miss, explained:** run ___, probability ___. Why the model thought it was safe: ...

## 5. Fairness lesson from the people demo

One or two sentences: what did the lender example show, and what does it mean for any model you deploy?

> ...

## 6. Recommendation

- **Trust the model on:** ...
- **Don't rely on it for:** ...
- **Do instead or as well:** ...
- **Watch after deployment (Day 4 monitoring):** ...
