"""
Linguistic quality metrics: word count, lexical diversity (MATTR), and
phrase repetition (Distinct-2), comparing real vs. synthetic free text.

Complements the existing semantic/distributional metrics in
source/metrics/free_text_metrics.py (Frechet/MMD on TF-IDF/SVD embeddings)
and source/metrics/run_llm_text_metrics.py (BERTScore/MAUVE) -- those
measure meaning and distinguishability; these measure observable surface
properties (length, vocabulary richness, repetition) that those metrics do
not directly capture. Motivated by recent findings that LLM-generated text
can be semantically fine while being less lexically/syntactically diverse
than real text (Guo et al. 2025; Shaib et al. 2024).

Produces, with no new LLM generation (everything is already on disk):
  1. A main free-text-linguistic-quality result: real vs. synthetic, all
     three metrics, across the 6 free-text datasets x 3 k=5 generator
     configs.
  2. A retrieval-ablation extension: k=5 vs. k=0 (Gfl only, the only
     generator with k=0 data), word count + lexical diversity only (no
     Distinct-2 in the ablation, by explicit scope decision).

Reporting convention: raw real and synthetic values side by side (no
ratios, gaps, or percent differences), for direct, easy-to-read comparison.
Unlike the structured-data fidelity metrics, NIFTSy's free-text generation
is not fit on a training split -- retrieval only reads individual examples
at generation time -- so there is no train/holdout calibration step here:
every real document is used as the single "real" reference.

Usage:
    python -m source.metrics.run_linguistic_quality
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from source.metrics.free_text_metrics import _clean_text_series

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "metrics.yml"
METADATA_ROOT = REPO_ROOT / ".metadata"
OUTPUT_DIR = REPO_ROOT / "reports" / "linguistic_quality"

DATASETS = ["adult_with_three", "Airbnb", "consumer_complaint", "fake_job_postings",
            "Kickstarter", "netflix_titles"]
MAIN_MODELS = ["gemini-3.1-flash-lite", "gpt-6-luna", "Qwen/Qwen2.5-14B-Instruct"]
K0_MODEL = "gemini-3.1-flash-lite_k0"

MATTR_WINDOW = 100
N_SEEDS = 5
MAX_ROWS = 20_000


def _safe_model_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "model"


def _load_metadata(dataset_name: str) -> dict[str, Any] | None:
    path = METADATA_ROOT / dataset_name / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@dataclass
class LinguisticQualitySummary:
    text_column: str
    variant: str
    n_real: int
    n_synthetic: int
    token_budget: int
    mattr_window: int
    n_seeds: int
    budget_limited_by: str
    # word count (whitespace-split, matches the existing max_words-prompting convention)
    wc_mean_real: float
    wc_mean_synthetic: float
    # MATTR (regex word tokenizer, fixed token budget, averaged over seeds)
    mattr_real: float
    mattr_synthetic: float
    mattr_std_real: float
    mattr_std_synthetic: float
    # Distinct-2 (same fixed token budget, within-document bigrams only)
    distinct2_real: float
    distinct2_synthetic: float
    distinct2_std_real: float
    distinct2_std_synthetic: float


_LEXICAL_TOKEN_RE = re.compile(r"\w+(?:['\-]\w+)*")


def tokenize_lexical(text: str) -> list[str]:
    """Regex word tokenizer for MATTR/Distinct-2 -- deliberately NOT whitespace-split,
    to avoid inflating apparent diversity on punctuation/JSON-like fields (e.g. Airbnb
    amenities lists). Word-count uses whitespace-split instead, for consistency with
    the existing max_words-prompting convention (run_full.py:_word_count_series)."""
    return _LEXICAL_TOKEN_RE.findall(text.lower())


def word_counts(series: pd.Series) -> np.ndarray:
    return series.str.split().str.len().to_numpy(dtype=float)


def tokenize_corpus(series: pd.Series) -> list[list[str]]:
    return [tokenize_lexical(t) for t in series.tolist()]


def sample_token_budget(docs: list[list[str]], budget: int, seed: int) -> list[list[str]]:
    """Shuffle whole documents, append until the token budget is reached, truncating
    the last document to fit exactly. Preserves document boundaries (needed by
    within-document Distinct-2)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    sampled: list[list[str]] = []
    total = 0
    for i in order:
        doc = docs[i]
        if not doc:
            continue
        remaining = budget - total
        if remaining <= 0:
            break
        if len(doc) > remaining:
            sampled.append(doc[:remaining])
            total += remaining
            break
        sampled.append(doc)
        total += len(doc)
    return sampled


def mattr(sampled_docs: list[list[str]], window: int) -> float:
    tokens = [t for doc in sampled_docs for t in doc]
    n = len(tokens)
    if n == 0:
        return float("nan")
    w = min(window, n)
    if n < w:
        w = n
    if w <= 0:
        return float("nan")
    if n == w:
        return len(set(tokens)) / w

    from collections import Counter
    counts: Counter[str] = Counter(tokens[:w])
    ratios = [len(counts) / w]
    for i in range(w, n):
        incoming = tokens[i]
        outgoing = tokens[i - w]
        counts[incoming] += 1
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        ratios.append(len(counts) / w)
    return float(np.mean(ratios))


def distinct_n(sampled_docs: list[list[str]], n: int = 2) -> float:
    total = 0
    unique: set[tuple[str, ...]] = set()
    for doc in sampled_docs:
        if len(doc) < n:
            continue
        for i in range(len(doc) - n + 1):
            gram = tuple(doc[i:i + n])
            unique.add(gram)
            total += 1
    if total == 0:
        return float("nan")
    return len(unique) / total


def _seeded_mattr_distinct2(docs: list[list[str]], budget: int, window: int, seeds: range) -> tuple[float, float, float, float]:
    mattr_vals, distinct_vals = [], []
    for seed in seeds:
        sampled = sample_token_budget(docs, budget, seed)
        mattr_vals.append(mattr(sampled, window))
        distinct_vals.append(distinct_n(sampled, 2))
    return (float(np.nanmean(mattr_vals)), float(np.nanstd(mattr_vals)),
            float(np.nanmean(distinct_vals)), float(np.nanstd(distinct_vals)))


def compute_token_budget(token_docs_by_corpus: dict[str, list[list[str]]]) -> tuple[int, str]:
    totals = {name: sum(len(d) for d in docs) for name, docs in token_docs_by_corpus.items()}
    limiting = min(totals, key=lambda k: totals[k])
    return totals[limiting], limiting


def evaluate_linguistic_quality(
    *,
    text_column: str,
    variant: str,
    real_text: pd.Series,
    synthetic_text: pd.Series,
    token_budget: int,
    budget_limited_by: str,
    real_lex_docs: list[list[str]],
    synthetic_lex_docs: list[list[str]],
    mattr_window: int = MATTR_WINDOW,
    seeds: range = range(N_SEEDS),
) -> dict[str, Any]:
    wc_re = word_counts(real_text)
    wc_sy = word_counts(synthetic_text)

    mattr_re, mattr_re_std, d2_re, d2_re_std = _seeded_mattr_distinct2(
        real_lex_docs, token_budget, mattr_window, seeds)
    mattr_sy, mattr_sy_std, d2_sy, d2_sy_std = _seeded_mattr_distinct2(
        synthetic_lex_docs, token_budget, mattr_window, seeds)

    summary = LinguisticQualitySummary(
        text_column=text_column,
        variant=variant,
        n_real=int(len(real_text)),
        n_synthetic=int(len(synthetic_text)),
        token_budget=token_budget,
        mattr_window=mattr_window,
        n_seeds=len(list(seeds)),
        budget_limited_by=budget_limited_by,
        wc_mean_real=float(np.mean(wc_re)),
        wc_mean_synthetic=float(np.mean(wc_sy)),
        mattr_real=mattr_re, mattr_synthetic=mattr_sy,
        mattr_std_real=mattr_re_std, mattr_std_synthetic=mattr_sy_std,
        distinct2_real=d2_re, distinct2_synthetic=d2_sy,
        distinct2_std_real=d2_re_std, distinct2_std_synthetic=d2_sy_std,
    )
    return asdict(summary)


def _load_capped(path: Path, max_rows: int, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state).reset_index(drop=True)
    return df


def _variant_synthetic_path(cfg: dict, dataset_name: str, model: str) -> Path:
    template = cfg.get("synthetic_data_path", "data/processed/{dataset_name}_{generator_model}.csv")
    safe_model = _safe_model_name(model)
    return REPO_ROOT / template.format(dataset_name=dataset_name, generator_model=safe_model)


def process_dataset(dataset_name: str, cfg: dict, random_state: int, holdout_fraction: float) -> dict[str, Any]:
    print(f"=== {dataset_name} ===", flush=True)
    metadata = _load_metadata(dataset_name)
    text_columns = [str(c) for c in (metadata.get("free_text_columns") or [])] if metadata else []
    if not text_columns:
        print(f"  no free-text columns in metadata, skipping")
        return {}

    real_template = cfg.get("real_data_path", "data/raw/{dataset_name}.csv")
    real_path = REPO_ROOT / real_template.format(dataset_name=dataset_name)
    real_df_full = _load_capped(real_path, MAX_ROWS, random_state)
    # Use only the held-out split as "real" (smaller, and keeps this comparison
    # on data the generator's retrieval step never reads), discarding the
    # training split -- there is no train/holdout calibration here.
    _, real_df = train_test_split(real_df_full, test_size=holdout_fraction, random_state=random_state)

    variants: dict[str, pd.DataFrame] = {}
    for model in MAIN_MODELS + [K0_MODEL]:
        syn_path = _variant_synthetic_path(cfg, dataset_name, model)
        if not syn_path.exists():
            print(f"  [skip] variant '{model}': file not found at {syn_path}")
            continue
        variants[model] = _load_capped(syn_path, MAX_ROWS, random_state)

    results: dict[str, Any] = {}
    for col in text_columns:
        if col not in real_df.columns:
            print(f"  [skip] column '{col}' not in real data")
            continue

        real_text = _clean_text_series(real_df[col])
        real_lex = tokenize_corpus(real_text)

        syn_texts: dict[str, pd.Series] = {}
        syn_lex: dict[str, list[list[str]]] = {}
        for model, syn_df in variants.items():
            df_use = syn_df
            if col not in df_use.columns:
                continue
            t = _clean_text_series(df_use[col])
            if t.empty:
                continue
            syn_texts[model] = t
            syn_lex[model] = tokenize_corpus(t)

        if not syn_texts:
            print(f"  [skip] column '{col}': no synthetic variant has this column")
            continue

        budget_corpora = {"real": real_lex, **syn_lex}
        token_budget, limiting = compute_token_budget(budget_corpora)
        print(f"  column '{col}': token_budget={token_budget} (limited by {limiting}), "
              f"variants={list(syn_texts.keys())}")

        col_results: dict[str, Any] = {}
        for model in syn_texts:
            col_results[model] = evaluate_linguistic_quality(
                text_column=col,
                variant=model,
                real_text=real_text,
                synthetic_text=syn_texts[model],
                token_budget=token_budget,
                budget_limited_by=limiting,
                real_lex_docs=real_lex,
                synthetic_lex_docs=syn_lex[model],
            )
        results[col] = col_results

    return results


# ---- Dataset/column display metadata for the LaTeX tables ------------------

_DS_MACRO = {
    "adult_with_three": r"\dsAdult", "Airbnb": r"\dsAirbnb",
    "consumer_complaint": r"\dsComplaint", "fake_job_postings": r"\dsJobs",
    "Kickstarter": r"\dsKick", "netflix_titles": r"\dsNetflix",
}
_COL_ORDER = [
    ("adult_with_three", "job_description"), ("adult_with_three", "daily_tasks"),
    ("adult_with_three", "work_snippet"), ("Airbnb", "name"), ("Airbnb", "amenities"),
    ("consumer_complaint", "consumer_complaint_narrative"), ("fake_job_postings", "description"),
    ("Kickstarter", "name"), ("netflix_titles", "title"), ("netflix_titles", "cast"),
    ("netflix_titles", "description"),
]


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    random_state = int(cfg.get("random_state", 42))

    holdout_fraction = float(cfg.get("holdout_fraction", 0.2))
    all_results: dict[str, Any] = {}
    for ds in DATASETS:
        all_results[ds] = process_dataset(ds, cfg, random_state, holdout_fraction)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
