"""Prefect retraining flow for Machine Guard.

    validate data -> check drift -> retrain -> evaluate -> promote only if it passes the gate

Usage (from the repo root):
    python flows/retrain_flow.py                                          # next-quarter batch
    python flows/retrain_flow.py --batch data/incoming/messy_batch.csv    # stops at validation
    python flows/retrain_flow.py --model logistic_regression --class-weight none   # weak: gate says no

What it does:
    1. Validates the base training data and the new labelled batch with the Pandera schema.
    2. Runs the Evidently drift check on the batch. No drift: stop, keep the champion.
    3. Retrains on the training split of the base data plus the new batch, using build_pipeline.
    4. Grades the new model on the fixed course test set (the same exam as the quality gate).
    5. Logs and registers it in MLflow as a new version with the alias "challenger".
       Only if recall passes the gate does it also get the alias "champion".

It trains through machine_guard.features.build_pipeline and the MLflow API directly, not through
`python -m machine_guard.train`, so the flow does not depend on train.py's command-line options.
Promotion only moves the alias. Shipping the new champion is a human step (docs/runbook.md):
export it to models/, commit, and let CI test, gate and build the image.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so "python flows/retrain_flow.py" finds machine_guard

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import pandera as pa  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402

from machine_guard.drift import check_drift  # noqa: E402
from machine_guard.features import (  # noqa: E402
    RAW_FEATURES, REGISTERED_MODEL_NAME, SEED, TARGET, build_pipeline, load_data, split,
)
from machine_guard.quality_gate import RECALL_GATE, evaluate, passes_gate  # noqa: E402
from machine_guard.validate import summarise, validate  # noqa: E402

EXPERIMENT = "machine-guard"


def md5(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@task
def validate_data(path: str) -> pd.DataFrame:
    """Stop the flow if the file breaks the schema (Celsius, gaps, unknown types, bad labels)."""
    logger = get_run_logger()
    df = pd.read_csv(path)
    try:
        validate(df, require_target=True)
    except pa.errors.SchemaErrors as err:
        summary = summarise(err)
        logger.error("Validation FAILED for %s:\n%s", path, summary.to_string(index=False))
        # A short message instead of Pandera's very long one; the table above has the details.
        raise ValueError(f"{path} failed validation ({int(summary['rows'].sum())} problems). "
                         "Nothing was retrained.") from None
    logger.info("Validation passed: %s (%s rows)", path, f"{len(df):,}")
    return df


@task
def detect_drift(reference: pd.DataFrame, batch: pd.DataFrame) -> dict:
    _, summary = check_drift(reference, batch)
    get_run_logger().info("Drift: %s; drifted columns: %s",
                          summary["drift_detected"], summary["drifted_columns"] or "none")
    return summary


@task
def retrain(train: pd.DataFrame, model: str, class_weight: Optional[str]):
    get_run_logger().info("Training %s (class_weight=%s) on %s rows", model, class_weight, f"{len(train):,}")
    return build_pipeline(model, class_weight).fit(train[RAW_FEATURES], train[TARGET])


@task
def evaluate_model(pipe, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    metrics = evaluate(pipe, X_test, y_test)
    get_run_logger().info("Test set: recall %.3f, precision %.3f, missed %d, false alarms %d",
                          metrics["recall"], metrics["precision"],
                          metrics["missed_failures"], metrics["false_alarms"])
    return metrics


@task
def register_and_promote(pipe, metrics: dict, params: dict, tags: dict, gate: float) -> dict:
    """Log the run, register a new version as challenger, and make it champion only if it passes."""
    logger = get_run_logger()
    client = MlflowClient()
    with mlflow.start_run(run_name="retrain-flow") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.sklearn.log_model(pipe, "model", registered_model_name=REGISTERED_MODEL_NAME)
    version = max(int(v.version) for v in
                  client.search_model_versions(f"run_id='{run.info.run_id}'"))
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", version)

    # TODO (Lab 3, step 2): decide with the same rule as CI. One line, using passes_gate().
    raise NotImplementedError("Write the promotion decision in register_and_promote()")
    if promoted:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)
        logger.info("PROMOTED: version %s is now @champion (recall %.3f >= gate %.2f)",
                    version, metrics["recall"], gate)
    else:
        logger.warning("NOT PROMOTED: version %s stays @challenger (recall %.3f < gate %.2f). "
                       "The champion is unchanged.", version, metrics["recall"], gate)
    return {"version": version, "run_id": run.info.run_id, "promoted": promoted}


@flow(name="machine-guard-retrain", log_prints=True)
def retrain_flow(
    batch_path: str = "data/incoming/next_quarter.csv",
    data_path: str = "data/raw/ai4i2020.csv",
    model: str = "random_forest",
    class_weight: Optional[str] = "balanced",
    retrain_without_drift: bool = False,
    gate: float = RECALL_GATE,
) -> dict:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    # 1. Validate: bad data never reaches training.
    base = validate_data(data_path)
    batch = validate_data(batch_path)

    # 2. Drift: retrain only when the world has changed (or when asked to).
    X_train, X_test, _, y_test = split(base)
    drift = detect_drift(base, batch)
    if not drift["drift_detected"] and not retrain_without_drift:
        print("No drift detected: nothing to retrain. The champion stays.")
        return {"retrained": False, "drift": drift}

    # 3. Retrain on old training rows plus the new batch. The test rows stay out.
    train = pd.concat([base.loc[X_train.index], batch], ignore_index=True)
    pipe = retrain(train, model, class_weight)

    # 4. Evaluate on the fixed course test set.
    metrics = evaluate_model(pipe, X_test, y_test)

    # 5. Register, and promote only if it passes the gate.
    params = {"model": model, "class_weight": str(class_weight), "engineered": True, "seed": SEED,
              "batch": batch_path, "train_rows": len(train)}
    tags = {"git_commit": git_commit(), "data_md5": md5(data_path), "batch_md5": md5(batch_path),
            "trigger": "drift" if drift["drift_detected"] else "manual"}
    result = register_and_promote(pipe, metrics, params, tags, gate)

    if result["promoted"]:
        print("Next: python -m machine_guard.export_model, then commit models/ and push "
              "(CI tests, gates and builds the new image).")
    return {"retrained": True, "drift": drift, "metrics": metrics, **result}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the Machine Guard retraining flow once.")
    parser.add_argument("--batch", default="data/incoming/next_quarter.csv")
    parser.add_argument("--data", default="data/raw/ai4i2020.csv")
    parser.add_argument("--model", default="random_forest",
                        choices=["random_forest", "decision_tree", "logistic_regression"])
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--force", action="store_true", help="retrain even if no drift is found")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    retrain_flow(
        batch_path=args.batch,
        data_path=args.data,
        model=args.model,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        retrain_without_drift=args.force,
    )
