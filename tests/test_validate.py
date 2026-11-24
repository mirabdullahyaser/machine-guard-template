"""Unit tests for the Pandera schema: good data passes, each planted problem is caught."""
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from machine_guard.validate import validate


def failed_checks(df: pd.DataFrame) -> set:
    """Run validation and return the set of (column, check) pairs that failed."""
    with pytest.raises(pa.errors.SchemaErrors) as err:
        validate(df)
    return set(zip(err.value.failure_cases["column"], err.value.failure_cases["check"]))


def test_original_data_passes(data):
    validate(data, require_target=True)


def test_extra_columns_are_allowed(sample):
    validate(sample.assign(comment="night shift"))


def test_celsius_air_temperature_is_rejected(sample):
    # TODO (Lab 1, step 3): turn the first 10 air temperatures into Celsius (subtract 273.15),
    # then assert that ("air_temp_k", "in_range(250.0, 350.0)") is in failed_checks(sample).
    pytest.skip("TODO: write this test")


def test_missing_torque_is_rejected(sample):
    # TODO (Lab 1, step 3): blank out one torque reading with np.nan,
    # then assert that ("torque_nm", "not_nullable") is in failed_checks(sample).
    pytest.skip("TODO: write this test")


def test_unknown_product_type_is_rejected(sample):
    sample.loc[0, "type"] = "X"
    assert ("type", "isin(['L', 'M', 'H'])") in failed_checks(sample)


def test_missing_column_is_rejected(sample):
    checks = failed_checks(sample.drop(columns=["tool_wear_min"]))
    assert any(check == "column_in_dataframe" for _, check in checks)


def test_all_problems_reported_at_once(sample):
    sample.loc[0, "air_temp_k"] = 25.0
    sample.loc[1, "torque_nm"] = np.nan
    columns = {col for col, _ in failed_checks(sample)}
    assert {"air_temp_k", "torque_nm"} <= columns


def test_training_data_needs_a_valid_label(sample):
    sample.loc[0, "machine_failure"] = 2
    with pytest.raises(pa.errors.SchemaErrors):
        validate(sample, require_target=True)
