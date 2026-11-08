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
VALID_STATES = {"failed", "successful", "canceled", "live", "undefined", "suspended"}


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    df = normalize_object_columns(df)

    # A small number of rows have an unescaped comma in `name`, which shifts
    # every later column one position left (e.g. `category` ends up holding
    # half the name, and `state` ends up holding a number instead of one of
    # its known categories). These rows are identifiable by an invalid
    # `state` value; dropping them also restores the correct numeric dtypes
    # for `goal`, `pledged`, and `backers`.
    df = df[df["state"].isin(VALID_STATES)].copy()
    df["goal"] = pd.to_numeric(df["goal"])
    df["pledged"] = pd.to_numeric(df["pledged"])
    df["backers"] = pd.to_numeric(df["backers"]).astype(int)

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
