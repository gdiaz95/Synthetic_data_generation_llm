from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import POTENTIAL_DIR, RAW_DIR, normalize_object_columns, run_preprocessing

INPUT_FILE = POTENTIAL_DIR / "english_teaching.csv"
OUTPUT_FILE = RAW_DIR / "english_teaching.csv"
# Student_ID is a row identifier, not a feature.
DROP_COLUMNS = ["Student_ID"]


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_object_columns(df)
    return df.drop(columns=DROP_COLUMNS)


def main() -> Path:
    return run_preprocessing("english_teaching", INPUT_FILE, OUTPUT_FILE, preprocess_columns)


if __name__ == "__main__":
    main()
