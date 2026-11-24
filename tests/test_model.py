"""Tests for the committed model file, models/model.joblib (written by `python -m machine_guard.export_model`).

These are behaviour tests: they don't check exact numbers, they check that the model behaves
like a sensible machine-failure model. CI runs them on every push.
"""
import os
from pathlib import Path

import joblib
import pytest

from machine_guard.features import RAW_FEATURES

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        if os.getenv("CI"):
            pytest.fail(f"{MODEL_PATH} is missing: export and commit the champion model")
        pytest.skip(f"{MODEL_PATH} not found (run python -m machine_guard.export_model)")
    return joblib.load(MODEL_PATH)


def test_model_gives_one_probability_per_row(model, sample):
    proba = model.predict_proba(sample[RAW_FEATURES])[:, 1]
    assert len(proba) == len(sample)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_more_strain_means_more_risk(model, healthy_frame):
    """Directional test: high torque on a worn tool must look riskier than a fresh, easy run."""
    strained = healthy_frame.assign(torque_nm=70.0, tool_wear_min=230, rotational_speed_rpm=1300)
    p_healthy = model.predict_proba(healthy_frame)[0, 1]
    p_strained = model.predict_proba(strained)[0, 1]
    assert p_strained > p_healthy


def test_column_order_does_not_matter(model, healthy_frame):
    """Invariance test: the pipeline selects columns by name, not by position."""
    shuffled = healthy_frame[list(reversed(RAW_FEATURES))]
    assert model.predict_proba(shuffled)[0, 1] == pytest.approx(model.predict_proba(healthy_frame)[0, 1])
