from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
POTENTIAL_DIR = DATA_DIR / "potential"
RAW_DIR = DATA_DIR / "raw"

Transform = Callable[[pd.DataFrame], pd.DataFrame]


def safe_decode(line: bytes) -> str:
    """Decode a CSV line using the first compatible encoding."""
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return line.decode(encoding)
        except UnicodeDecodeError:
            continue
    return line.decode("utf-8", errors="replace")



def fix_mojibake(text):
    if text is None or pd.isna(text):
        return text
    if not isinstance(text, str):
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



def load_csv(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_file}")

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



def normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].map(fix_mojibake)
    return df



def run_preprocessing(name: str, input_file: Path, output_file: Path, transform: Transform) -> Path:
    df = load_csv(input_file)
    df = transform(df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"[{name}] Saved raw dataset to: {output_file}")
    print(f"[{name}] Shape: {df.shape}")
    print(f"[{name}] Columns: {df.columns.tolist()}")
    return output_file
