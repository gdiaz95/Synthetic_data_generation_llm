import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo

load_dotenv()

MODEL = "gemini-3-flash-preview"
MAX_WORDS = 80
TEMPERATURE = float("0.9")
TOP_P = float("0.95")
TOP_K = int("20")
THINKING_BUDGET = int("2048")  # medium reasoning budget
SLEEP_BETWEEN_CALLS_SEC = float("0.05")

BATCH_SIZE = 50
EXISTING_JOB_DESC_CSV = "../data/small/adult_with_job_description_small.csv"
OUTPUT_CSV = "../data/small/adult_with_both.csv"


def load_source_dataframe() -> tuple[pd.DataFrame, bool]:
    if os.path.exists(EXISTING_JOB_DESC_CSV):
        print(f"✅ Data already there. Loading existing file: {EXISTING_JOB_DESC_CSV}")
        return pd.read_csv(EXISTING_JOB_DESC_CSV).replace("?", pd.NA), True

    print("ℹ️ Pre-generated CSV not found. Fetching Adult dataset from UCI repo (fetch_ucirepo)...")
    data = fetch_ucirepo(id=2)
    df_loaded = pd.concat([data.data.features, data.data.targets], axis=1)
    df_loaded["income"] = df_loaded["income"].str.strip().str.replace(".", "", regex=False)
    return df_loaded.replace("?", pd.NA), False


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


def to_jsonable(x):
    if pd.isna(x):
        return None
    if hasattr(x, "item"):
        return x.item()
    return x


def _build_fields(row: pd.Series) -> dict:
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
    return {k: v for k, v in fields.items() if v is not None}


def build_job_description_batch_prompt(rows_payload) -> str:
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


def build_daily_tasks_batch_prompt(rows_payload) -> str:
    return f"""
You are generating a realistic free-text field for census/employment-style records.

For EACH input record, output ONE JSON object PER LINE (JSONL), with keys:
- "id": integer (copy input id)
- "daily_tasks": string (25–{MAX_WORDS} words)

Hard constraints:
- MAX {MAX_WORDS} words (do not exceed).
- Write in third person from an outside perspective (e.g., "They", "This worker", "The employee").
- Do NOT write in first person (no "I", "my", "me").
- Describe a short, realistic summary of routine daily tasks.
- Keep consistent with structured fields and provided job description context.
- Do NOT include names, employer names, exact addresses, or company locations.
- Output ONLY JSON lines. No extra commentary.

INPUT RECORDS (JSON):
{json.dumps(rows_payload, ensure_ascii=False)}
""".strip()


def _generate_jsonl_batch(rows_payload, prompt_builder, output_key: str, max_retries: int = 6) -> dict:
    prompt = prompt_builder(rows_payload)

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents={"text": prompt},
                config={
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "thinking_config": {"thinking_budget": THINKING_BUDGET},
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
                    content = _cap_words(str(obj[output_key]), MAX_WORDS)
                    out[rid] = content
                except Exception:
                    continue

            return out

        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"[warn] Gemini call failed (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait}s")
            time.sleep(wait)

    return {}


def generate_job_descriptions_for_batch(rows_payload, max_retries: int = 6) -> dict:
    return _generate_jsonl_batch(rows_payload, build_job_description_batch_prompt, "job_description", max_retries)


def generate_daily_tasks_for_batch(rows_payload, max_retries: int = 6) -> dict:
    return _generate_jsonl_batch(rows_payload, build_daily_tasks_batch_prompt, "daily_tasks", max_retries)


def main():
    df, loaded_existing_job_desc_file = load_source_dataframe()
    df_small = df.head(df.shape[0]).copy()
    print(f"Processing {df_small.shape[0]} rows...")
    print(f"Output path: {OUTPUT_CSV}")

    has_job_desc_col = "job_description" in df_small.columns
    has_non_empty_job_desc = has_job_desc_col and df_small["job_description"].fillna("").astype(str).str.strip().ne("").any()
    needs_job_desc_generation = not has_non_empty_job_desc

    job_desc = {}
    if has_non_empty_job_desc:
        print("✅ Existing non-empty 'job_description' detected. Reusing it and skipping regeneration.")
        job_desc = {int(i): str(v).strip() for i, v in df_small["job_description"].fillna("").items()}
    else:
        print("ℹ️ 'job_description' missing/empty. Generating job_description first, then daily_tasks.")

    daily_tasks = {}

    indices = list(df_small.index)
    batches = [indices[i:i + BATCH_SIZE] for i in range(0, len(indices), BATCH_SIZE)]

    for batch_ids in tqdm(batches, desc="Batches", unit="batch", dynamic_ncols=True):
        rows_payload = []
        for idx in batch_ids:
            fields = _build_fields(df_small.loc[idx])
            rows_payload.append({"id": int(idx), "fields": fields})

        if needs_job_desc_generation:
            print("[stage] Generating job_description batch...")
            parsed_job_desc = generate_job_descriptions_for_batch(rows_payload)
            for idx in batch_ids:
                rid = int(idx)
                job_desc[rid] = parsed_job_desc.get(rid, "")

        rows_with_job_context = []
        for item in rows_payload:
            rid = int(item["id"])
            rows_with_job_context.append(
                {
                    "id": rid,
                    "fields": item["fields"],
                    "job_description": job_desc.get(rid, ""),
                }
            )

        print("[stage] Generating daily_tasks batch...")
        parsed_daily_tasks = generate_daily_tasks_for_batch(rows_with_job_context)
        for idx in batch_ids:
            rid = int(idx)
            daily_tasks[rid] = parsed_daily_tasks.get(rid, "")

        if SLEEP_BETWEEN_CALLS_SEC > 0:
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    df_small["job_description"] = [job_desc.get(int(i), "") for i in df_small.index]
    df_small["daily_tasks"] = [daily_tasks.get(int(i), "") for i in df_small.index]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_small.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
