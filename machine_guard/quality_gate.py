"""Model quality gate: stop the pipeline if the model misses too many failures.

Usage (from the repo root):
    python -m machine_guard.quality_gate                                   # checks models/model.joblib
    python -m machine_guard.quality_gate --model-uri "models:/machine-guard@champion"
    python -m machine_guard.quality_gate --gate 0.80                       # stricter gate
    python -m machine_guard.quality_gate --batch data/incoming/next_quarter.csv   # grade on new labelled data

Exit code 0 means the model may be promoted or shipped. Exit code 1 means it may not, and CI turns red.

By default the gate grades on the same exam every time: the course test set, i.e. the 20% of
data/raw/ai4i2020.csv held out by machine_guard.features.split (seed 42, stratified).
Grade a model on the same data file it was trained on, or test rows leak into training.
With --batch it grades on every row of a new labelled batch instead (is the model still fit for it?).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from machine_guard.features import RAW_FEATURES, TARGET, load_data, split

RECALL_GATE = 0.75            # recall on failures must be at least this (course contract)
DECISION_THRESHOLD = 0.5      # probability above which we predict "failure"
COST_MISSED = 20              # Day 1 cost ratio: one missed breakdown costs 20 false alarms
COST_FALSE_ALARM = 1


def evaluate(model, X: pd.DataFrame, y: pd.Series, threshold: float = DECISION_THRESHOLD) -> dict:
    """The course metrics for one model on one dataset (same names as the MLflow runs)."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "caught_failures": int(tp),
        "missed_failures": int(fn),
        "false_alarms": int(fp),
        "cost": int(fn * COST_MISSED + fp * COST_FALSE_ALARM),
    }


def passes_gate(metrics: dict, gate: float = RECALL_GATE) -> bool:
    """True if the model may be promoted: recall on failures is at least the gate."""
    # TODO (Lab 1, step 4): one line. Return True when metrics["recall"] is at least `gate`.
    raise NotImplementedError("Write the gate decision in passes_gate()")


def load_model(model_path: str | None = None, model_uri: str | None = None):
    if model_uri:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        return mlflow.sklearn.load_model(model_uri)
    return joblib.load(model_path)


def data_version_warning(model_path: str, data_path: str) -> str | None:
    """Warn if the data file is not the one the model was trained on (models/model_meta.json)."""
    meta_path = Path(model_path).with_name("model_meta.json")
    if not meta_path.exists() or not Path(data_path).exists():
        return None
    expected = json.loads(meta_path.read_text()).get("data_md5")
    actual = hashlib.md5(Path(data_path).read_bytes()).hexdigest()
    if expected and expected != actual:
        return (f"WARNING: {data_path} (md5 {actual[:8]}) is not the data this model was trained on "
                f"(md5 {expected[:8]}). Test rows may have been in its training data.")
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fail (exit 1) if recall on failures is below the gate.")
    parser.add_argument("--model-path", default="models/model.joblib", help="a saved scikit-learn pipeline")
    parser.add_argument("--model-uri", help='an MLflow model URI instead, e.g. "models:/machine-guard@champion"')
    parser.add_argument("--data", default="data/raw/ai4i2020.csv", help="data file whose test split is graded")
    parser.add_argument("--batch", help="grade on every row of this labelled CSV instead of the test split")
    parser.add_argument("--gate", type=float, default=RECALL_GATE, help="minimum recall on failures")
    parser.add_argument("--json", help="also write the metrics and verdict to this JSON file")
    args = parser.parse_args(argv)

    model = load_model(args.model_path, args.model_uri)
    if args.batch:
        batch = pd.read_csv(args.batch)
        X_test, y_test = batch[RAW_FEATURES], batch[TARGET]
    else:
        _, X_test, _, y_test = split(load_data(args.data))
    metrics = evaluate(model, X_test, y_test)
    passed = passes_gate(metrics, args.gate)

    source = args.model_uri or args.model_path
    print(f"Quality gate for {source}")
    warning = None if (args.model_uri or args.batch) else data_version_warning(args.model_path, args.data)
    if warning:
        print(f"  {warning}")
    graded_on = args.batch or "test set"
    print(f"  {graded_on}: {len(y_test):,} runs, {int(y_test.sum())} real failures")
    print(f"  recall    {metrics['recall']:.3f}   (gate: at least {args.gate:.2f})")
    print(f"  precision {metrics['precision']:.3f}")
    print(f"  caught {metrics['caught_failures']}, missed {metrics['missed_failures']}, "
          f"false alarms {metrics['false_alarms']}, cost {metrics['cost']}")
    print("PASSED: the model may be promoted" if passed else "FAILED: recall is below the gate, do not promote")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({**metrics, "gate": args.gate, "passed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
