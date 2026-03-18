import csv
from io import StringIO
from pathlib import Path

import pandas as pd

POTENTIAL_FILE = Path("data/potential/Airbnb.csv")
RAW_FILE = Path("data/raw/Airbnb.csv")
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


def safe_decode(line: bytes) -> str:
    """Decode a CSV line using the first compatible encoding."""
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return line.decode(encoding)
        except UnicodeDecodeError:
            continue
    return line.decode("utf-8", errors="replace")


def fix_mojibake(text: str | None) -> str | None:
    if text is None:
        return text

    suspicious = ("Â", "Ã", "Å", "â", "€", "œ", "�")
    if not any(char in text for char in suspicious):
        return text

    for source_encoding in ("latin1", "cp1252"):
        try:
            return text.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

    return text


def load_cleaned_dataframe(input_file: Path) -> pd.DataFrame:
    cleaned_buffer = StringIO()
    writer = csv.writer(cleaned_buffer)

    with input_file.open("rb") as infile:
        for raw_line in infile:
            decoded_line = safe_decode(raw_line)
            row = next(csv.reader([decoded_line]))
            cleaned_row = [fix_mojibake(field) for field in row]
            writer.writerow(cleaned_row)

    cleaned_buffer.seek(0)
    return pd.read_csv(cleaned_buffer, low_memory=False)


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype("string").str.strip()

    for column in ("amenities", "name"):
        if column in df.columns:
            df[column] = df[column].astype("string")

    return df


def main() -> None:
    if RAW_FILE.exists():
        print(f"Done. Raw dataset already exists at: {RAW_FILE}")
        return

    if not POTENTIAL_FILE.exists():
        raise FileNotFoundError(f"Missing input dataset: {POTENTIAL_FILE}")

    df = load_cleaned_dataframe(POTENTIAL_FILE)
    df = preprocess_columns(df)

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)

    print(f"Done. Saved raw dataset to: {RAW_FILE}")
    print("Columns:")
    print(df.columns.tolist())
    print("Shape:")
    print(df.shape)


if __name__ == "__main__":
    main()
