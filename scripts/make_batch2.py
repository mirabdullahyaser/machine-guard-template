"""Make "batch 2": 2,000 new, clean sensor runs from the same process as the original data.

Day 2, DVC lab. Usage (from the repo root):
    python scripts/make_batch2.py            # writes data/incoming/batch2.csv
    python scripts/make_batch2.py --append   # also appends it to data/raw/ai4i2020.csv ("version 2")

How the new runs are made (so the batch has the same distribution as the original, not copies of it):
- Each sensor reading is drawn from the original file's own values, with a little measurement noise,
  so the ranges and shapes match. Readings that move together are drawn together from the same
  original run: air and process temperature from one run; product type, torque, speed and tool wear
  from another.
- The labels are NOT copied. They come from the five failure rules documented with the dataset:
  tool wear failure (TWF), heat dissipation (HDF), power (PWF), overstrain (OSF) and random (RNF).
  A run fails if any rule fires. The rules reproduce the original's HDF, PWF and OSF flags exactly.
- Same seed, same file: running it twice gives byte-identical output, so the DVC hashes in the
  lab sheet match yours.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "ai4i2020.csv"
OUTPUT = REPO_ROOT / "data" / "incoming" / "batch2.csv"

N_ROWS = 2000
SEED = 2  # batch 2; the course seed 42 stays for model training

OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}  # tool wear x torque, minute-newton-metres


def make_batch(original: pd.DataFrame, n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    row = lambda: rng.integers(0, len(original), n)  # noqa: E731  random rows of the original

    # Readings that belong together are drawn together, from the same original run, plus a little noise.
    t = row()
    air = np.round(original["air_temp_k"].to_numpy()[t] + rng.normal(0, 0.2, n), 1)
    process = np.round(original["process_temp_k"].to_numpy()[t] + rng.normal(0, 0.2, n), 1)
    m = row()
    types = original["type"].to_numpy()[m]
    torque = np.round(np.clip(original["torque_nm"].to_numpy()[m] + rng.normal(0, 0.5, n), 3.8, None), 1)
    speed = np.round(original["rotational_speed_rpm"].to_numpy()[m] + rng.normal(0, 10, n)).astype(int)
    wear = original["tool_wear_min"].to_numpy()[m].astype(int)

    # Failure rules (AI4I 2020 documentation)
    power = torque * speed * 2 * np.pi / 60
    hdf = ((process - air) < 8.6) & (speed < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = wear * torque > pd.Series(types).map(OSF_LIMIT).to_numpy()
    # Tools wear out somewhere between 200 and 240 minutes; use the original's rate in that window.
    in_window = original["tool_wear_min"].between(200, 240)
    twf_rate = original.loc[in_window, "twf"].mean()
    twf = ((wear >= 200) & (wear <= 240)) & (rng.random(n) < twf_rate)
    rnf = rng.random(n) < 0.001

    modes = pd.DataFrame({"twf": twf, "hdf": hdf, "pwf": pwf, "osf": osf, "rnf": rnf}).astype(int)
    first_udi = int(original["udi"].max()) + 1
    serials = 60000 + np.arange(n)  # original serials stop at 57174, so no clashes

    batch = pd.DataFrame({
        "udi": np.arange(first_udi, first_udi + n),
        "product_id": [f"{t}{s}" for t, s in zip(types, serials)],
        "type": types,
        "air_temp_k": air,
        "process_temp_k": process,
        "rotational_speed_rpm": speed,
        "torque_nm": torque,
        "tool_wear_min": wear,
        "machine_failure": modes.max(axis=1),
    })
    return pd.concat([batch, modes], axis=1)[original.columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--append", action="store_true", help="also append the batch to data/raw/ai4i2020.csv")
    args = parser.parse_args()

    raw = pd.read_csv(RAW)
    original = raw[raw["udi"] <= 10000]  # always generate from version 1, even after appending
    batch = make_batch(original)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}: {len(batch):,} rows, "
          f"{batch['machine_failure'].sum()} failures ({batch['machine_failure'].mean():.2%})")

    if args.append:
        if raw["udi"].isin(batch["udi"]).any():
            print(f"{RAW.relative_to(REPO_ROOT)} already contains batch 2 ({len(raw):,} rows). Nothing appended.")
            return
        combined = pd.concat([raw, batch], ignore_index=True)
        combined.to_csv(RAW, index=False)
        print(f"Appended to {RAW.relative_to(REPO_ROOT)}: now {len(combined):,} rows, "
              f"{combined['machine_failure'].sum()} failures ({combined['machine_failure'].mean():.2%})")


if __name__ == "__main__":
    main()
