"""Data drift check with Evidently: does new data still look like the data the model learned from?

Usage (from the repo root):
    python -m machine_guard.drift --current data/incoming/next_quarter.csv   # drift: yes
    python -m machine_guard.drift --holdout                                  # control: our own test set, drift: no
    python -m machine_guard.drift --current logs/predictions.db              # what the live API has seen

Writes (all under reports/ by default):
    drift_report.html    the full Evidently report: open it in a browser
    drift_summary.json   the verdict and one line per column, for scripts and CI
    drift_summary.md     the same as a Markdown table, used as the body of the GitHub alert issue

Alert rule: the batch counts as drifted when at least DRIFT_SHARE_ALERT of the six model inputs
have drifted (2 of 6 columns). Evidently picks the test per column: with more than 1,000 reference
rows it uses the Wasserstein distance for numbers and the Jensen-Shannon distance for `type`,
and calls a column drifted when the distance is above 0.1.

Exit code 0 whatever the verdict, unless --fail-on-drift is given (then 1 on drift); 2 if there are no rows.
Evidently is pinned to 0.4.40 in requirements.txt: later releases moved these imports.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from machine_guard.features import NUMERIC_RAW, RAW_FEATURES, load_data, split

DRIFT_SHARE_ALERT = 0.3   # 2 of the 6 inputs drifting is enough to raise an alert
MIN_ROWS = 100            # below this, drift scores are mostly noise


def load_batch(path: str) -> pd.DataFrame:
    """Read a CSV batch, or the `predictions` table of the API's SQLite prediction log."""
    if path.endswith(".db"):
        with sqlite3.connect(path) as con:
            return pd.read_sql_query("SELECT * FROM predictions", con)
    return pd.read_csv(path)


def check_drift(reference: pd.DataFrame, current: pd.DataFrame, drift_share: float = DRIFT_SHARE_ALERT):
    """Compare the six model inputs. Returns (Evidently report, summary dict)."""
    if len(current) == 0:
        raise ValueError("The current data has no rows: nothing to compare.")
    # Same column types on both sides (an SQLite log can come back as text).
    reference = reference[RAW_FEATURES].astype({c: float for c in NUMERIC_RAW})
    current = current[RAW_FEATURES].astype({c: float for c in NUMERIC_RAW})
    report = Report(metrics=[DataDriftPreset(drift_share=drift_share)])
    report.run(reference_data=reference, current_data=current)

    results = report.as_dict()["metrics"]
    dataset = results[0]["result"]                      # DatasetDriftMetric
    by_column = results[1]["result"]["drift_by_columns"]  # DataDriftTable
    columns = {
        name: {
            "drifted": bool(col["drift_detected"]),
            "score": round(float(col["drift_score"]), 4),
            "test": col["stattest_name"],
            "threshold": col["stattest_threshold"],
        }
        for name, col in sorted(by_column.items())
    }
    summary = {
        "drift_detected": bool(dataset["dataset_drift"]),
        "drifted_columns": [c for c, v in columns.items() if v["drifted"]],
        "share_drifted": round(float(dataset["share_of_drifted_columns"]), 3),
        "alert_share": drift_share,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "columns": columns,
    }
    return report, summary


def to_markdown(summary: dict, current_name: str) -> str:
    verdict = "DRIFT DETECTED" if summary["drift_detected"] else "No drift"
    lines = [
        f"**{verdict}** in `{current_name}` ({summary['current_rows']:,} rows "
        f"vs {summary['reference_rows']:,} reference rows).",
        "",
        f"{len(summary['drifted_columns'])} of {len(summary['columns'])} inputs drifted "
        f"(alert when the share is at least {summary['alert_share']}).",
        "",
        "| Column | Drifted | Score | Test (threshold) |",
        "|---|---|---|---|",
    ]
    for name, col in summary["columns"].items():
        flag = "**yes**" if col["drifted"] else "no"
        lines.append(f"| `{name}` | {flag} | {col['score']} | {col['test']} ({col['threshold']}) |")
    lines += ["", "Next step: follow `docs/runbook.md`, section \"Drift alert\"."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare a batch with the training data and report drift.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--current", help="new data: a CSV file, or logs/predictions.db")
    source.add_argument("--holdout", action="store_true",
                        help="control run: compare the course test split with the training split")
    parser.add_argument("--reference", default="data/raw/ai4i2020.csv", help="the data the model learned from")
    parser.add_argument("--out-dir", default="reports", help="where to write the report and summaries")
    parser.add_argument("--drift-share", type=float, default=DRIFT_SHARE_ALERT)
    parser.add_argument("--fail-on-drift", action="store_true", help="exit code 1 when drift is detected")
    args = parser.parse_args(argv)

    if args.holdout:
        X_train, X_test, _, _ = split(load_data(args.reference))
        reference, current, name = X_train, X_test, "course test split (control)"
    else:
        reference, current, name = load_data(args.reference), load_batch(args.current), args.current

    if len(current) == 0:
        print(f"No rows in {name}: nothing to check. (Has the API logged any predictions yet?)")
        return 2
    if len(current) < MIN_ROWS:
        print(f"WARNING: only {len(current)} rows in {name}. With fewer than {MIN_ROWS}, drift scores are mostly noise.")
    report, summary = check_drift(reference, current, args.drift_share)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out / "drift_report.html"))
    (out / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "drift_summary.md").write_text(to_markdown(summary, name))

    print(f"Drift check: {name} vs {args.reference}")
    for col, v in summary["columns"].items():
        print(f"  {col:<22} {'DRIFT' if v['drifted'] else 'ok':<6} score {v['score']:<8} {v['test']}")
    verdict = "DRIFT DETECTED" if summary["drift_detected"] else "no drift"
    print(f"Result: {verdict} ({len(summary['drifted_columns'])} of {len(summary['columns'])} columns)")
    print(f"Report: {out / 'drift_report.html'}")
    return 1 if (args.fail_on_drift and summary["drift_detected"]) else 0


if __name__ == "__main__":
    sys.exit(main())
