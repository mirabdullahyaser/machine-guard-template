"""Make the two Day 4 incoming batches from the training data.

Usage (from the repo root):
    python scripts/simulate_next_quarter.py

Writes:
    data/incoming/next_quarter.csv  2,000 runs from "next quarter". Two things changed in the plant:
        1. A new tool supplier: the new tools last longer, so the tool-wear clock reads about 40% higher
           before the same amount of real wear (the link between tool_wear_min and failure has moved).
        2. The process temperature sensor was recalibrated and now reads 2.5 K higher.
       Every value is still inside VALID_RANGES, so data validation passes. Only a drift check notices.
    data/incoming/messy_batch.csv   500 runs with planted problems for the validation lab:
        - one production line (150 runs) sends air temperature in Celsius instead of kelvin;
        - 25 torque readings and 15 tool-wear readings are missing.

Both batches are drawn only from the training rows of the course split (machine_guard.features.split),
so the test set that the quality gate uses is never copied into new training data.
The script is deterministic: same seed, same files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so "python scripts/simulate_next_quarter.py" finds machine_guard

from machine_guard.features import NUMERIC_RAW, SEED, VALID_RANGES, load_data, split  # noqa: E402
INCOMING = REPO_ROOT / "data" / "incoming"

NEXT_QUARTER_ROWS = 2000
MESSY_ROWS = 500

TOOL_WEAR_FACTOR = 1.4        # new supplier: the clock reads 40% higher for the same real wear
PROCESS_TEMP_OFFSET_K = 2.5   # recalibrated sensor reads 2.5 K higher
CELSIUS_ROWS = 150            # messy batch: one line reports air temperature in Celsius
MISSING_TORQUE = 25
MISSING_TOOL_WEAR = 15


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    X_train, _, _, _ = split(df)
    return df.loc[X_train.index]


def make_next_quarter(train: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    batch = train.sample(n=NEXT_QUARTER_ROWS, replace=True, random_state=SEED).reset_index(drop=True)
    batch["udi"] = np.arange(20001, 20001 + len(batch))

    # Small measurement noise, so rows are not exact copies of training rows.
    batch["air_temp_k"] = (batch["air_temp_k"] + rng.normal(0, 0.1, len(batch))).round(1)
    batch["process_temp_k"] = (batch["process_temp_k"] + rng.normal(0, 0.1, len(batch))).round(1)
    batch["torque_nm"] = (batch["torque_nm"] + rng.normal(0, 0.3, len(batch))).clip(lower=0).round(1)

    # The two real changes.
    batch["tool_wear_min"] = (batch["tool_wear_min"] * TOOL_WEAR_FACTOR).round().astype(int)
    batch["process_temp_k"] = (batch["process_temp_k"] + PROCESS_TEMP_OFFSET_K).round(1)

    for col in NUMERIC_RAW:
        low, high = VALID_RANGES[col]
        assert batch[col].between(low, high).all(), f"{col} left VALID_RANGES: drift must stay valid"
    return batch


def make_messy(train: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    batch = train.sample(n=MESSY_ROWS, replace=False, random_state=SEED + 1).reset_index(drop=True)
    batch["udi"] = np.arange(30001, 30001 + len(batch))
    batch["torque_nm"] = batch["torque_nm"].astype(float)
    batch["tool_wear_min"] = batch["tool_wear_min"].astype(float)

    line4 = rng.choice(len(batch), CELSIUS_ROWS, replace=False)
    batch.loc[line4, "air_temp_k"] = (batch.loc[line4, "air_temp_k"] - 273.15).round(1)

    batch.loc[rng.choice(len(batch), MISSING_TORQUE, replace=False), "torque_nm"] = np.nan
    batch.loc[rng.choice(len(batch), MISSING_TOOL_WEAR, replace=False), "tool_wear_min"] = np.nan
    return batch


def describe(name: str, batch: pd.DataFrame, train: pd.DataFrame) -> None:
    print(f"\n{name}: {len(batch):,} rows, failure rate {batch['machine_failure'].mean():.2%}")
    print(f"  {'column':<22}{'training mean':>15}{'batch mean':>12}{'missing':>9}")
    for col in NUMERIC_RAW:
        print(f"  {col:<22}{train[col].mean():>15.1f}{batch[col].mean():>12.1f}{batch[col].isna().sum():>9}")


def main() -> int:
    df = load_data(str(REPO_ROOT / "data" / "raw" / "ai4i2020.csv"))
    train = training_rows(df)
    rng = np.random.default_rng(SEED)

    INCOMING.mkdir(parents=True, exist_ok=True)
    next_quarter = make_next_quarter(train, rng)
    messy = make_messy(train, rng)
    next_quarter.to_csv(INCOMING / "next_quarter.csv", index=False)
    messy.to_csv(INCOMING / "messy_batch.csv", index=False)

    describe("data/incoming/next_quarter.csv", next_quarter, train)
    describe("data/incoming/messy_batch.csv", messy, train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
