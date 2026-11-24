"""Download the AI4I 2020 Predictive Maintenance dataset and save it with course column names.

Usage (from the repo root):
    python scripts/download_data.py            # use the bundled copy if present, else download
    python scripts/download_data.py --force    # always download a fresh copy from UCI

Output: data/raw/ai4i2020.csv  (10,000 rows, snake_case columns)

Dataset credit: Stephan Matzka, "AI4I 2020 Predictive Maintenance Dataset",
UCI Machine Learning Repository, 2020. Licence CC BY 4.0. https://doi.org/10.24432/C5HS5C
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = REPO_ROOT / "data" / "raw" / "ai4i2020_original.csv"
OUTPUT = REPO_ROOT / "data" / "raw" / "ai4i2020.csv"

# One rename, used by every notebook and script in the course.
COLUMNS = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


def download_original() -> None:
    print(f"Downloading {UCI_URL} ...")
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        ORIGINAL.parent.mkdir(parents=True, exist_ok=True)
        ORIGINAL.write_bytes(zf.read(csv_name))
    print(f"Saved original file to {ORIGINAL.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download even if a local copy exists")
    args = parser.parse_args()

    if args.force or not ORIGINAL.exists():
        download_original()
    else:
        print(f"Using bundled copy {ORIGINAL.relative_to(REPO_ROOT)}")

    # utf-8-sig strips the byte-order mark at the start of the UCI file.
    df = pd.read_csv(ORIGINAL, encoding="utf-8-sig")
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        print(f"ERROR: unexpected file layout, missing columns: {sorted(missing)}")
        return 1

    df = df.rename(columns=COLUMNS)[list(COLUMNS.values())]
    df.to_csv(OUTPUT, index=False)

    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}: {len(df):,} rows, {df.shape[1]} columns")
    print(f"Failure rate: {df['machine_failure'].mean():.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
