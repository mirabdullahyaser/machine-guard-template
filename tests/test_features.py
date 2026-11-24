"""Unit tests for the shared feature code. Each test checks one small promise."""
import math

import pandas as pd

from machine_guard.features import ENGINEERED, RAW_FEATURES, add_features, build_pipeline, split


def test_power_is_torque_times_angular_speed(healthy_frame):
    out = add_features(healthy_frame)
    expected = 42.8 * 1551 * 2 * math.pi / 60
    assert math.isclose(out.loc[0, "power_w"], expected, rel_tol=1e-9)


def test_temp_diff_is_process_minus_air(healthy_frame):
    out = add_features(healthy_frame)
    assert math.isclose(out.loc[0, "temp_diff_k"], 308.6 - 298.1, rel_tol=1e-9)


def test_add_features_does_not_change_its_input(healthy_frame):
    before = healthy_frame.copy()
    add_features(healthy_frame)
    pd.testing.assert_frame_equal(healthy_frame, before)
    assert not set(ENGINEERED) & set(healthy_frame.columns)


def test_split_is_stratified_80_20(data):
    X_train, X_test, y_train, y_test = split(data)
    assert len(X_train) == 8000 and len(X_test) == 2000
    assert abs(y_train.mean() - y_test.mean()) < 0.002
    assert list(X_train.columns) == RAW_FEATURES


def test_split_is_reproducible(data):
    first = split(data)[1].index
    second = split(data)[1].index
    assert first.equals(second)


def test_pipeline_takes_raw_columns_and_returns_probabilities(sample):
    pipe = build_pipeline(n_estimators=10).fit(sample[RAW_FEATURES], sample["machine_failure"])
    proba = pipe.predict_proba(sample[RAW_FEATURES])[:, 1]
    assert proba.min() >= 0 and proba.max() <= 1
