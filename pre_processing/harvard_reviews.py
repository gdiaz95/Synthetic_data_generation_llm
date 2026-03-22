from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pandas as pd

from _common import POTENTIAL_DIR, RAW_DIR, normalize_object_columns, safe_decode

INPUT_FILE = POTENTIAL_DIR / "harvard_reviews.csv"
OUTPUT_FILE = RAW_DIR / "harvard_reviews.csv"
EXPECTED_COLUMNS = [
    "published_date",
    "published_platform",
    "rating",
    "type",
    "helpful_votes",
    "title",
    "text",
]
HEADER_LINE = '"' + '","'.join(EXPECTED_COLUMNS) + '"'


def extract_review_rows(raw_text: str) -> list[list[str]]:
    normalized_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    header_positions: list[int] = []
    search_start = 0

    while True:
        position = normalized_text.find(HEADER_LINE, search_start)
        if position == -1:
            break
        header_positions.append(position)
        search_start = position + len(HEADER_LINE)

    if not header_positions:
        header_positions = [0]

    rows: list[list[str]] = []
    segment_end_positions = header_positions[1:] + [len(normalized_text)]

    for segment_start, segment_end in zip(header_positions, segment_end_positions):
        segment = normalized_text[segment_start:segment_end].strip()
        if not segment:
            continue

        reader = csv.reader(StringIO(segment))
        try:
            header = next(reader)
        except StopIteration:
            continue

        normalized_header = [column.strip() for column in header]
        if normalized_header != EXPECTED_COLUMNS:
            continue

        for row in reader:
            if not row:
                continue
            if [value.strip() for value in row] == EXPECTED_COLUMNS:
                continue
            if len(row) != len(EXPECTED_COLUMNS):
                continue
            rows.append(row)

    return rows



def load_harvard_reviews(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_file}")

    raw_text = safe_decode(input_file.read_bytes())
    rows = extract_review_rows(raw_text)
    if not rows:
        raise ValueError(f"No valid Harvard review rows were found in: {input_file}")

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    return df.drop_duplicates(ignore_index=True)



def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_object_columns(df)

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype("string").str.strip()

    df["published_date"] = pd.to_datetime(
        df["published_date"].astype("string").str.slice(0, 10), errors="coerce"
    )
    df["published_year"] = df["published_date"].dt.year
    df["published_month"] = df["published_date"].dt.month
    df["published_day"] = df["published_date"].dt.day
    df = df.drop(columns=["published_date"])

    for column in ("rating", "helpful_votes"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df



def main() -> Path:
    if OUTPUT_FILE.exists():
        print(f"[harvard_reviews] Skipping pre-processing because output already exists: {OUTPUT_FILE}")
        return OUTPUT_FILE

    df = load_harvard_reviews(INPUT_FILE)
    df = preprocess_columns(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"[harvard_reviews] Saved raw dataset to: {OUTPUT_FILE}")
    print(f"[harvard_reviews] Shape: {df.shape}")
    print(f"[harvard_reviews] Columns: {df.columns.tolist()}")
    return OUTPUT_FILE


if __name__ == "__main__":
    main()
