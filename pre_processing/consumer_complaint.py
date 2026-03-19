from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import POTENTIAL_DIR, RAW_DIR, normalize_object_columns, run_preprocessing

INPUT_FILE = POTENTIAL_DIR / "consumer_complaint.csv"
OUTPUT_FILE = RAW_DIR / "consumer_complaint.csv"
DROP_COLUMNS = [
    "sub_issue",
    "zip_code",
    "complaint_id",
    "date_sent_to_company",
    "company_public_response",
]


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(r"[^\w]+", "_", regex=True).str.strip("_")
    )
    df = normalize_object_columns(df)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
        df["received_year"] = df["date_received"].dt.year
        df["received_month"] = df["date_received"].dt.month
        df["received_day"] = df["date_received"].dt.day
        df = df.drop(columns=["date_received"])

    return df


def main() -> Path:
    return run_preprocessing("consumer_complaint", INPUT_FILE, OUTPUT_FILE, preprocess_columns)


if __name__ == "__main__":
    main()
