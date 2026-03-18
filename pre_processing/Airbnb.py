import csv
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


input_file = "data/potential/Airbnb.csv"
output_file = "data/potential/Airbnb_fixed.csv"


def safe_decode(line):
    """
    Try multiple encodings safely.
    """
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            return line.decode(enc)
        except UnicodeDecodeError:
            continue
    # last resort: replace bad chars
    return line.decode("utf-8", errors="replace")


def fix_mojibake(text):
    if text is None:
        return text

    suspicious = ["Â", "Ã", "Å", "â", "€", "œ", "�"]
    if not any(s in text for s in suspicious):
        return text

    try:
        return text.encode("latin1").decode("utf-8")
    except:
        pass

    try:
        return text.encode("cp1252").decode("utf-8")
    except:
        pass

    return text


with open(input_file, "rb") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:

    writer = csv.writer(outfile)

    for line_number, raw_line in enumerate(infile, start=1):

        decoded_line = safe_decode(raw_line)

        row = next(csv.reader([decoded_line]))

        fixed_row = [fix_mojibake(field) for field in row]

        writer.writerow(fixed_row)

print("Done. File cleaned safely.")

INPUT_FILE = "data/potential/Airbnb_fixed.csv"
OUTPUT_FILE = "data/raw/airbnb.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Columns to drop
drop_cols = [
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

df = df.drop(columns=drop_cols, errors="ignore")

# Minimal cleaning (keep everything as-is, just strip whitespace)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("string").str.strip()

# Ensure amenities and name are strings (important for later text modeling)
df["amenities"] = df["amenities"].astype("string")
df["name"] = df["name"].astype("string")

# Save
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved dataset to: {OUTPUT_FILE}")
print("\nColumns:")
print(df.columns.tolist())
print("\nShape:")
print(df.shape)