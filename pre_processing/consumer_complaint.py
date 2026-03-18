import io
import pandas as pd

INPUT_FILE = "data/potential/consumer_complaint.csv"
OUTPUT_REPORT = "dataset_report.txt"
OUTPUT_FILE = "data/raw/consumer_complaint.csv"

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

# Normalize column names once at the beginning
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[^\w]+", "_", regex=True)
    .str.strip("_")
)

cols_to_drop = [
    "sub_issue",
    "zip_code",
    "complaint_id",
    "date_sent_to_company",
    "company_public_response",
]

df = df.drop(columns=cols_to_drop, errors="ignore")

if "date_received" in df.columns:
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["received_year"] = df["date_received"].dt.year
    df["received_month"] = df["date_received"].dt.month
    df["received_day"] = df["date_received"].dt.day
    df = df.drop(columns=["date_received"])

print(df.head())
print(df.shape)
print(df.columns)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")