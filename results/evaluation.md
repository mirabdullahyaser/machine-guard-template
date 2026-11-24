# Machine Guard: Day 1 model evaluation

**Name:** 
**Date:** 
**Notebook:** `notebooks/03_first_model.ipynb`

## 1. Best model

Which model, with which settings and features?

...

## 2. How good is it? (test set: 2,000 runs, 68 real failures)

| | Baseline (always no failure) | My best model |
|---|---|---|
| Accuracy | 96.6% | |
| Precision on failures | 0% | |
| Recall on failures | 0% | |
| Failures caught (of 68) | 0 | |
| Missed breakdowns | 68 | |
| False alarms | 0 | |

## 3. What this means on the shop floor

In plain words, for a maintenance manager. For example: "Out of 68 real breakdowns it caught N. For every 10 alarms it raises, about P are real."

...

## 4. One maintenance recommendation

A specific action backed by something you saw in the data (for example a tool wear or torque level at which to plan a check).

...

## 5. Would I trust it on the shop floor?

Yes / No / Yes, if... and one reason. What would you need to see before you let it trigger real work orders?

...
