# Experiments, recorded by hand

Fill one row **after every model you train**, before you try the next idea.
Use the same split every time: `test_size=0.2`, `stratify=y`, `random_state=42` (68 failures in the test set).

| Run | Time | Model | Settings | Features | Accuracy | Precision | Recall | F1 | Missed breakdowns | False alarms | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | Baseline (always no failure) | `strategy="most_frequent"` | none | | | | | | | |
| 2 | | Logistic regression | `max_iter=1000`, scaled | 8 base columns | | | | | | | |
| 3 | | Decision tree | `max_depth=6` | 8 base columns | | | | | | | |
| 4 | | | | | | | | | | | |
| 5 | | | | | | | | | | | |
| 6 | | | | | | | | | | | |
| 7 | | | | | | | | | | | |
| 8 | | | | | | | | | | | |

## Three questions about this table (answer at 16:15)

1. Could someone else rerun your run 4 and get exactly the same numbers from this table alone? What is missing?
2. Which run's model would you deploy? Where is that model saved right now?
3. How long did filling this table take you, and how many typos did you catch?

Answers:

1. ...
2. ...
3. ...
