from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import POTENTIAL_DIR, RAW_DIR, normalize_object_columns, run_preprocessing

INPUT_FILE = POTENTIAL_DIR / "student_satisfaction.csv"
OUTPUT_FILE = RAW_DIR / "student_satisfaction.csv"


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_object_columns(df)



def main() -> Path:
    return run_preprocessing("student_satisfaction", INPUT_FILE, OUTPUT_FILE, preprocess_columns)


if __name__ == "__main__":
    main()
