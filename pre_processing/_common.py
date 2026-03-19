from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
POTENTIAL_DIR = DATA_DIR / "potential"
RAW_DIR = DATA_DIR / "raw"

Transform = Callable[[pd.DataFrame], pd.DataFrame]


def safe_decode(raw_content: bytes) -> str:
    """Decode file contents using the first compatible encoding."""
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return raw_content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_content.decode("utf-8", errors="replace")


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


def load_csv(input_file: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_file}")

    raw_content = input_file.read_bytes()
    decoded_text = safe_decode(raw_content)

    default_kwargs: dict[str, Any] = {}
    engine = read_csv_kwargs.get("engine")
    if engine != "python":
        default_kwargs["low_memory"] = False

    default_kwargs.update(read_csv_kwargs)
    return pd.read_csv(StringIO(decoded_text), **default_kwargs)


def normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].map(fix_mojibake)
    return df


def run_preprocessing(
    name: str,
    input_file: Path,
    output_file: Path,
    transform: Transform,
    **read_csv_kwargs: Any,
) -> Path:
    if output_file.exists():
        print(f"[{name}] Skipping pre-processing because output already exists: {output_file}")
        return output_file

    df = load_csv(input_file, **read_csv_kwargs)
    df = transform(df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"[{name}] Saved raw dataset to: {output_file}")
    print(f"[{name}] Shape: {df.shape}")
    print(f"[{name}] Columns: {df.columns.tolist()}")
    return output_file
