from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import POTENTIAL_DIR, RAW_DIR, run_preprocessing

INPUT_FILE = POTENTIAL_DIR / "Airbnb.csv"
OUTPUT_FILE = RAW_DIR / "Airbnb.csv"
DROP_COLUMNS = [
    "listing_id",
    "host_id",
    "host_since",
    "host_location",
    "district",
    "city",
    "latitude",
    "longitude",
    "room_type",
]


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype("string").str.strip()

    for column in ("amenities", "name"):
        if column in df.columns:
            df[column] = df[column].astype("string")

    return df


def main() -> Path:
    return run_preprocessing("Airbnb", INPUT_FILE, OUTPUT_FILE, preprocess_columns)


if __name__ == "__main__":
    main()
