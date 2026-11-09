from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import POTENTIAL_DIR, RAW_DIR, normalize_object_columns, run_preprocessing

INPUT_FILE = POTENTIAL_DIR / "Spotify.csv"
OUTPUT_FILE = RAW_DIR / "Spotify.csv"
# track_id is a row identifier; the unnamed column is the source CSV's row index.
DROP_COLUMNS = ["track_id"]


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    df = normalize_object_columns(df)
    unnamed_cols = [column for column in df.columns if column.startswith("unnamed")]
    return df.drop(columns=DROP_COLUMNS + unnamed_cols)


def main() -> Path:
    return run_preprocessing("Spotify", INPUT_FILE, OUTPUT_FILE, preprocess_columns)


if __name__ == "__main__":
    main()
