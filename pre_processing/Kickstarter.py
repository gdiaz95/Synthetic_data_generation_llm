from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import POTENTIAL_DIR, RAW_DIR, normalize_object_columns, run_preprocessing

INPUT_FILE = POTENTIAL_DIR / "Kickstarter.csv"
OUTPUT_FILE = RAW_DIR / "Kickstarter.csv"
DROP_COLUMNS = [
    "main_category",
    "deadline",
    "id",
    "usd_pledged",
]


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    df = normalize_object_columns(df)

    unnamed_cols = [column for column in df.columns if column.startswith("unnamed")]
    df = df.drop(columns=DROP_COLUMNS + unnamed_cols, errors="ignore")

    if "launched" in df.columns:
        df["launched"] = pd.to_datetime(df["launched"], errors="coerce")
        df["launch_year"] = df["launched"].dt.year
        df["launch_month"] = df["launched"].dt.month
        df["launch_day"] = df["launched"].dt.day
        df = df.drop(columns=["launched"])

    return df


def main() -> Path:
    return run_preprocessing("Kickstarter", INPUT_FILE, OUTPUT_FILE, preprocess_columns)


if __name__ == "__main__":
    main()
