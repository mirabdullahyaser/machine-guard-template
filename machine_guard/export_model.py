"""Export the champion model from the MLflow registry to plain files the API can load.

Run from the repo root:

    python -m machine_guard.export_model                 # champion from MLflow -> models/
    python -m machine_guard.export_model --alias challenger
    python -m machine_guard.export_model --fallback-train # ONLY if Day 2's registry is missing

It writes two files, which are committed to Git:

    models/model.joblib      the scikit-learn pipeline (raw sensor columns in, probability out)
    models/model_meta.json   model_version, run_id, data_md5, recall (+ where it came from)

Why export at all? The Docker image, Render, CI and the Kubernetes cluster cannot see the
MLflow database in your codespace. A plain file inside the repo goes everywhere the code goes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from machine_guard.features import REGISTERED_MODEL_NAME, SEED, build_pipeline, load_data, split

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_DATA = "data/raw/ai4i2020.csv"
THRESHOLD = 0.5


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sqlite_file(tracking_uri: str) -> Path | None:
    """The database file behind a sqlite:/// URI, or None for other kinds of URI."""
    prefix = "sqlite:///"
    return Path(tracking_uri[len(prefix):]) if tracking_uri.startswith(prefix) else None


def export_from_registry(tracking_uri: str, alias: str):
    """Load models:/machine-guard@<alias>. Returns (pipeline, meta). Raises RuntimeError if not possible."""
    db = sqlite_file(tracking_uri)
    # Check first: connecting to a missing sqlite file would quietly create an empty database.
    if db is not None and not db.exists():
        raise RuntimeError(f"No MLflow database at {db.resolve()}. Run this from the repo root, "
                           "after Day 2's training script has registered a model.")

    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    except MlflowException as err:
        raise RuntimeError(f"MLflow has no model '{REGISTERED_MODEL_NAME}' with alias '{alias}'. "
                           f"({err.message})") from err

    uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
    # The sklearn flavour gives back the real Pipeline, so the API does not need MLflow installed.
    pipeline = mlflow.sklearn.load_model(uri)
    run = client.get_run(version.run_id)
    meta = {
        # e.g. "machine-guard-model-3": registry version 3. (Not the image tag v1/v2: that is the app.)
        "model_version": f"{REGISTERED_MODEL_NAME}-model-{version.version}",
        "run_id": version.run_id,
        "data_md5": run.data.tags.get("data_md5"),
        "recall": round(run.data.metrics["recall"], 4) if "recall" in run.data.metrics else None,
        "source": f"mlflow registry: {uri}",
        "registered_version": int(version.version),
        "alias": alias,
    }
    return pipeline, meta


def train_fallback(data_path: str):
    """FALLBACK ONLY: train the Day 1 champion settings directly, with no MLflow.

    Use this when Day 2's registry is missing, so Day 3 is never blocked. The model is the same
    recipe as the contract's champion (random forest, balanced, engineered features), but nothing
    tracks it: no run id, no registry version. Re-export from the registry when you can.
    """
    from sklearn.metrics import recall_score

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split(df)
    pipeline = build_pipeline(model="random_forest", class_weight="balanced", engineered=True)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, (proba >= THRESHOLD).astype(int))
    meta = {
        "model_version": "fallback-untracked",
        "run_id": None,
        "data_md5": file_md5(data_path),
        "recall": round(float(recall), 4),
        "source": "FALLBACK: trained by export_model.py --fallback-train, not from the MLflow registry",
        "seed": SEED,
    }
    return pipeline, meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export the registered model to models/ for serving.")
    parser.add_argument("--alias", default="champion", help="registry alias to export (default: champion)")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--data", default=DEFAULT_DATA, help="training data (used by --fallback-train)")
    parser.add_argument(
        "--fallback-train", action="store_true",
        help="FALLBACK: if the MLflow registry or alias is missing, train a model with build_pipeline "
             "instead. Clearly labelled in model_meta.json. Use only if Day 2's registry is not available.")
    args = parser.parse_args(argv)

    try:
        pipeline, meta = export_from_registry(args.tracking_uri, args.alias)
    except RuntimeError as err:
        if not args.fallback_train:
            print(f"ERROR: {err}", file=sys.stderr)
            print("If Day 2's registry is not available, re-run with --fallback-train.", file=sys.stderr)
            return 1
        print(f"WARNING: {err}", file=sys.stderr)
        print("WARNING: --fallback-train given: training a model with build_pipeline instead. "
              "It is NOT tracked in MLflow.", file=sys.stderr)
        pipeline, meta = train_fallback(args.data)

    meta.update({
        "threshold": THRESHOLD,
        "sklearn_version": sklearn.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "model.joblib")
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    size_mb = (out / "model.joblib").stat().st_size / 1e6
    print(f"Exported {meta['model_version']} -> {out / 'model.joblib'} ({size_mb:.1f} MB)")
    print(f"recall={meta['recall']}  run_id={meta['run_id']}  data_md5={meta['data_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
