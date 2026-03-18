import io
import pandas as pd

INPUT_FILE = "data/potential/fake_job_postings.csv"
OUTPUT_REPORT = "dataset_report.txt"
OUTPUT_FILE = "data/raw/fake_job_postings.csv"

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


with open(INPUT_FILE, "rb") as f:
    lines = f.readlines()

decoded_lines = [safe_decode(line) for line in lines]
csv_text = "".join(decoded_lines)

df = pd.read_csv(io.StringIO(csv_text), low_memory=False)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].map(fix_mojibake)

print(df.head())
print(df.shape)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")