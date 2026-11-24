"""Shared definitions for Machine Guard: columns, valid ranges, features and the model pipeline.

Every later piece of the project imports from here, so the training script (Day 2), the API (Day 3),
the data validation, tests, drift check and retraining flow (Day 4) all agree on the same columns and rules.
Change something here once and it changes everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEST_SIZE = 0.2
TARGET = "machine_failure"

# What a sensor reading looks like when it arrives: the model's raw input.
RAW_FEATURES = [
    "type",
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
NUMERIC_RAW = RAW_FEATURES[1:]
ENGINEERED = ["power_w", "temp_diff_k"]

# Never features: ids, and the failure modes (target leakage: you only know them after the failure).
ID_COLUMNS = ["udi", "product_id"]
LEAKAGE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]

TYPE_VALUES = ["L", "M", "H"]

# Physically sensible ranges, shared by the API's input validation (Day 3) and the data validation
# schema (Day 4). Wider than the training data on purpose: they reject impossible values
# (a temperature in Celsius, a negative speed), not unusual ones. Unusual ones are drift's job.
VALID_RANGES = {
    "air_temp_k": (250.0, 350.0),
    "process_temp_k": (250.0, 360.0),
    "rotational_speed_rpm": (0.0, 5000.0),
    "torque_nm": (0.0, 150.0),
    "tool_wear_min": (0.0, 400.0),
}

REGISTERED_MODEL_NAME = "machine-guard"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with power_w and temp_diff_k added (the two Day 1 feature ideas)."""
    out = df.copy()
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * np.pi / 60
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    return out


def load_data(path: str = "data/raw/ai4i2020.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def split(df: pd.DataFrame):
    """Stratified 80/20 split with the course seed. Returns X_train, X_test, y_train, y_test."""
    X = df[RAW_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def make_estimator(model: str = "random_forest", class_weight: str | None = "balanced", **params):
    """The model families used in the course, with the Day 1 settings as defaults."""
    if model == "random_forest":
        defaults = dict(n_estimators=200, min_samples_leaf=5, n_jobs=-1)
        defaults.update(params)
        return RandomForestClassifier(class_weight=class_weight, random_state=SEED, **defaults)
    if model == "decision_tree":
        defaults = dict(max_depth=6)
        defaults.update(params)
        return DecisionTreeClassifier(class_weight=class_weight, random_state=SEED, **defaults)
    if model == "logistic_regression":
        defaults = dict(max_iter=1000)
        defaults.update(params)
        return LogisticRegression(class_weight=class_weight, random_state=SEED, **defaults)
    raise ValueError(f"Unknown model: {model}")


def build_pipeline(model: str = "random_forest", class_weight: str | None = "balanced",
                   engineered: bool = True, **params) -> Pipeline:
    """Raw sensor columns in, failure probability out.

    The pipeline does everything: adds the engineered features, one-hot encodes `type`, scales
    numbers, then fits the model. So whoever uses the saved model (the API, the drift check)
    only needs to pass the six RAW_FEATURES columns.
    """
    numeric = NUMERIC_RAW + (ENGINEERED if engineered else [])
    steps = []
    if engineered:
        steps.append(("add_features", FunctionTransformer(add_features)))
    steps.append((
        "prepare",
        ColumnTransformer([
            ("type", OneHotEncoder(categories=[TYPE_VALUES], handle_unknown="ignore"), ["type"]),
            ("numbers", StandardScaler(), numeric),
        ]),
    ))
    steps.append(("model", make_estimator(model, class_weight, **params)))
    return Pipeline(steps)
