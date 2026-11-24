"""Data validation for Machine Guard: a Pandera schema built from the shared rules in features.py.

Usage (from the repo root):
    python -m machine_guard.validate data/raw/ai4i2020.csv              # passes, exit code 0
    python -m machine_guard.validate data/incoming/messy_batch.csv      # fails, exit code 1

The schema checks what a sensor reading must look like before anything trains on it or scores it:
the six model inputs exist, have no gaps, `type` is L, M or H, and every number is inside
VALID_RANGES (the same ranges the API uses). It rejects impossible values, such as a temperature
in Celsius. It does not judge unusual-but-possible values: that is the drift check's job.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import pandera as pa

from machine_guard.features import NUMERIC_RAW, TARGET, TYPE_VALUES, VALID_RANGES


def build_schema(require_target: bool = False) -> pa.DataFrameSchema:
    """One Pandera column rule per model input, plus the label when we train on the data."""
    columns = {
        "type": pa.Column(str, pa.Check.isin(TYPE_VALUES), nullable=False),
    }
    for name in NUMERIC_RAW:
        low, high = VALID_RANGES[name]
        # TODO (Lab 1, step 2): add a rule for this column to `columns`. It must:
        #   - hold numbers: float, with coerce=True (so 1551 and 1551.0 are both fine)
        #   - be inside the valid range: pa.Check.in_range(low, high)
        #   - have no gaps: nullable=False
        # Use the "type" rule above as your model.
        pass
    columns[TARGET] = pa.Column(
        int, pa.Check.isin([0, 1]), nullable=False, coerce=True, required=require_target,
    )
    # strict=False: extra columns (ids, failure modes) are allowed; the model never sees them.
    return pa.DataFrameSchema(columns, strict=False, name="machine_guard_sensor_data")


SCHEMA = build_schema()


def validate(df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """Return the validated data, or raise pandera.errors.SchemaErrors listing every problem.

    lazy=True collects all failures instead of stopping at the first one.
    """
    return build_schema(require_target).validate(df, lazy=True)


def summarise(err: pa.errors.SchemaErrors) -> pd.DataFrame:
    """One line per (column, check) with the number of bad rows and a few example values."""
    cases = err.failure_cases
    return (
        cases.groupby(["column", "check"], dropna=False)
        .agg(rows=("index", "size"), examples=("failure_case", lambda s: list(s.drop_duplicates()[:3])))
        .reset_index()
        .sort_values("rows", ascending=False)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a CSV of sensor readings.")
    parser.add_argument("path", help="CSV file to check, for example data/incoming/messy_batch.csv")
    parser.add_argument("--require-target", action="store_true",
                        help="the file must also have a valid machine_failure label (training data)")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.path)
    try:
        validate(df, require_target=args.require_target)
    except pa.errors.SchemaErrors as err:
        summary = summarise(err)
        print(f"FAILED: {args.path} ({len(df):,} rows), {int(summary['rows'].sum())} problems found\n")
        with pd.option_context("display.width", 120, "display.max_colwidth", 60):
            print(summary.to_string(index=False))
        return 1
    print(f"PASSED: {args.path} ({len(df):,} rows) matches the Machine Guard schema")
    return 0


if __name__ == "__main__":
    sys.exit(main())
