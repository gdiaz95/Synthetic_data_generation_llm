import pandas as pd
import sys
import os
from ucimlrepo import fetch_ucirepo
from google import genai
import re
import time
from tqdm import tqdm
from dotenv import load_dotenv
import json

load_dotenv()

MODEL = "gemini-3-flash-preview"
MAX_WORDS = 80
TEMPERATURE = float("0.9")
TOP_P = float("0.95")
TOP_K = int("20")
SLEEP_BETWEEN_CALLS_SEC = float("0.05")

BATCH_SIZE = 50  # <-- NEW


# ----------------------------
# Data load
# ----------------------------
data = fetch_ucirepo(id=2)
df = pd.concat([data.data.features, data.data.targets], axis=1)
df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
df = df.replace("?", pd.NA)

# ----------------------------
# Gemini client
# ----------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not loaded. Check .env")

client = genai.Client(api_key=api_key)


def _cap_words(text: str, max_words: int = MAX_WORDS) -> str:
    words = re.findall(r"\S+", (text or "").strip())
    if len(words) <= max_words:
        return " ".join(words).strip()
    return " ".join(words[:max_words]).strip()


# ---- NEW: numpy/pandas -> JSON safe ----
def to_jsonable(x):
    if pd.isna(x):
        return None
    if hasattr(x, "item"):  # numpy scalars: int64/float64/bool_
        return x.item()
    return x


# ---- NEW: build batch prompt ----
def build_batch_prompt(rows_payload) -> str:
    return f"""
You are generating a realistic free-text field that could be collected on a census/employment form.

For EACH input record, output ONE JSON object PER LINE (JSONL), with keys:
- "id": integer (copy input id)
- "job_description": string (40–{MAX_WORDS} words)

Hard constraints:
- MAX {MAX_WORDS} words (do not exceed).
- Do NOT include names, employer names, exact addresses, or specific company locations.
- Do NOT mention salary numbers or explain the income label.
- Do NOT restate fields as a list; write natural prose.
- Adequate the tone to be like a self-reported survey answer, not a formal job description. 
- Adapt tone to what you consider a person in that job salary status would give.
- Stay consistent with the structured fields; if uncertain, keep it generic.
- Output ONLY JSON lines. No extra commentary.

INPUT RECORDS (JSON):
{json.dumps(rows_payload, ensure_ascii=False)}
""".strip()


# ---- NEW: call Gemini once per batch and parse JSONL ----
def generate_job_descriptions_for_batch(rows_payload, max_retries: int = 6) -> dict:
    prompt = build_batch_prompt(rows_payload)

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents={"text": prompt},
                config={
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                },
            )
            text = (resp.text or "").strip()

            out = {}
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rid = int(obj["id"])
                    jd = _cap_words(str(obj["job_description"]), MAX_WORDS)
                    out[rid] = jd
                except Exception:
                    continue

            return out

        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"[warn] Gemini call failed (attempt {attempt+1}/{max_retries}): {e}. Waiting {wait}s")
            time.sleep(wait)

    return {}  # if all retries fail


def main():
    number_of_rows = df.shape[0]  # <-- change to 100 to test quickly   
    df_small = df.head(number_of_rows) # <-- change to df.head(10) to test quickly
    out_path = "../data/small/adult_with_job_description.csv"
    print(f"Processing {df_small.shape[0]} rows...")
    print(f"Output path: {out_path}")
    job_desc = {}

    indices = list(df_small.index)
    batches = [indices[i:i + BATCH_SIZE] for i in range(0, len(indices), BATCH_SIZE)]

    for batch_ids in tqdm(batches, desc="Batches", unit="batch", dynamic_ncols=True):
        rows_payload = []
        for idx in batch_ids:
            row = df_small.loc[idx]
            fields = {
                "age": to_jsonable(row.get("age")),
                "workclass": to_jsonable(row.get("workclass")),
                "education": to_jsonable(row.get("education")),
                "marital_status": to_jsonable(row.get("marital-status")),
                "occupation": to_jsonable(row.get("occupation")),
                "relationship": to_jsonable(row.get("relationship")),
                "sex": to_jsonable(row.get("sex")),
                "hours_per_week": to_jsonable(row.get("hours-per-week")),
                "native_country": to_jsonable(row.get("native-country")),
            }
            # drop missing
            fields = {k: v for k, v in fields.items() if v is not None}
            rows_payload.append({"id": int(idx), "fields": fields})

        parsed = generate_job_descriptions_for_batch(rows_payload)

        # assign + print as they come in
        for idx in batch_ids:
            idx = int(idx)
            jd = parsed.get(idx, "")
            job_desc[idx] = jd
            if jd:
                print(f"\n--- row {idx} ---\n{jd}\n" + "-" * 40)

        if SLEEP_BETWEEN_CALLS_SEC > 0:
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    df_small = df_small.copy()
    df_small["job_description"] = [job_desc.get(int(i), "") for i in df_small.index]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_small.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()