from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import POTENTIAL_DIR, RAW_DIR, normalize_object_columns, run_preprocessing

INPUT_FILE = POTENTIAL_DIR / "fake_job_postings.csv"
OUTPUT_FILE = RAW_DIR / "fake_job_postings.csv"
# job_id is a row identifier; company_profile, requirements and benefits are long
# free text that is not modeled as free text (only `description` is), so the
# structured synthesizer would copy them verbatim from real rows.
DROP_COLUMNS = ["job_id", "company_profile", "requirements", "benefits"]


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_object_columns(df)
    return df.drop(columns=DROP_COLUMNS)



def main() -> Path:
    return run_preprocessing("fake_job_postings", INPUT_FILE, OUTPUT_FILE, preprocess_columns)


if __name__ == "__main__":
    main()
