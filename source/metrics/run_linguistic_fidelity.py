"""
Linguistic fidelity metrics: word count, lexical diversity (MATTR), and
phrase repetition (Distinct-2), comparing real vs. synthetic free text.

Complements the existing semantic/distributional metrics in
source/metrics/free_text_metrics.py (Frechet/MMD on TF-IDF/SVD embeddings)
and source/pipelines/run_llm_text_metrics.py (BERTScore/MAUVE) -- those
measure meaning and distinguishability; these measure observable surface
properties (length, vocabulary richness, repetition) that those metrics do
not directly capture. Motivated by recent findings that LLM-generated text
can be semantically fine while being less lexically/syntactically diverse
than real text (Guo et al. 2025; Shaib et al. 2024).

Produces, with no new LLM generation (everything is already on disk):
  1. A main free-text-linguistic-fidelity result: real vs. synthetic, all
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
    python -m source.metrics.run_linguistic_fidelity
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
OUTPUT_DIR = REPO_ROOT / "reports" / "linguistic_fidelity"
TEX_OUTPUT_PATH = REPO_ROOT / "writing" / "linguistic_fidelity.tex"

DATASETS = ["adult_with_three", "Airbnb", "consumer_complaint", "fake_job_postings",
            "Kickstarter", "netflix_titles"]
MAIN_MODELS = ["gemini-3.1-flash-lite-preview", "gpt-4o-mini", "Qwen/Qwen2.5-14B-Instruct"]
K0_MODEL = "gemini-3.1-flash-lite-preview_k0"
K0_BASE_MODEL = "gemini-3.1-flash-lite-preview"

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
class LinguisticFidelitySummary:
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


def evaluate_linguistic_fidelity(
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

    summary = LinguisticFidelitySummary(
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


PATCH_DIR = REPO_ROOT / "data" / "processed" / "patches"


def _patch_path(dataset_name: str, model: str, col: str) -> Path:
    """Small (e.g. 300-row) standalone samples generated for a single free-text column
    when the official released synthetic file for that (dataset, generator) is missing
    it -- e.g. consumer_complaint's gpt-4o-mini file has no consumer_complaint_narrative
    column at all. Same convention already used for the joint-fidelity appendix tables."""
    safe_model = _safe_model_name(model)
    return PATCH_DIR / f"{dataset_name}_{safe_model}_{col}_small.csv"


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
                patch_path = _patch_path(dataset_name, model, col)
                if not patch_path.exists():
                    continue
                df_use = _load_capped(patch_path, MAX_ROWS, random_state)
                print(f"  [patch] variant '{model}', column '{col}': using {patch_path.name} "
                      f"({len(df_use)} rows; official file has no such column)")
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
            col_results[model] = evaluate_linguistic_fidelity(
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


def _fmt(x: float, nd: int = 2) -> str:
    return "---" if x != x else f"{x:.{nd}f}"


def _mean_over_variants(col_results: dict[str, Any], models: list[str], field: str) -> float:
    vals = [col_results[m][field] for m in models if m in col_results]
    return float(np.mean(vals)) if vals else float("nan")


def build_main_table(all_results: dict[str, Any]) -> str:
    rows = []
    for ds, col in _COL_ORDER:
        col_results = all_results.get(ds, {}).get(col, {})
        if not col_results:
            continue
        present = [m for m in MAIN_MODELS if m in col_results]
        any_variant = col_results[present[0]] if present else next(iter(col_results.values()))
        wc_real = any_variant["wc_mean_real"]
        wc_syn = _mean_over_variants(col_results, present, "wc_mean_synthetic")
        mattr_real = any_variant["mattr_real"]
        mattr_syn = _mean_over_variants(col_results, present, "mattr_synthetic")
        d2_real = any_variant["distinct2_real"]
        d2_syn = _mean_over_variants(col_results, present, "distinct2_synthetic")
        label = f"{_DS_MACRO[ds]} ({col.replace('_', chr(92)+'_')})"
        rows.append(
            f"{label:32} & {_fmt(wc_real, 1)} & {_fmt(wc_syn, 1)} & "
            f"{_fmt(mattr_real, 3)} & {_fmt(mattr_syn, 3)} & "
            f"{_fmt(d2_real, 3)} & {_fmt(d2_syn, 3)} \\\\"
        )
    body = "\n".join(rows)
    return r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{\textbf{Free-text linguistic fidelity.} Avg.\ words is the mean word count per entry. MATTR (moving-average type-token ratio) measures lexical diversity as the fraction of unique word types within a sliding window of text; higher values indicate a more varied vocabulary. Distinct-2 measures phrase repetition as the fraction of unique word bigrams within a document; higher values indicate less repetitive text. Each synthetic cell reports the mean across \Gfl, \GPT, and \Qfourteen.}
\label{tab:linguistic-fidelity}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Words} & \multicolumn{2}{c}{MATTR} & \multicolumn{2}{c}{Distinct-2} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\textbf{Dataset (column)} & \textbf{Real} & \textbf{Synth} & \textbf{Real} & \textbf{Synth} & \textbf{Real} & \textbf{Synth} \\ \midrule
""" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def build_ablation_table(all_results: dict[str, Any]) -> str:
    ds_order = DATASETS
    cols_by_ds = {
        "adult_with_three": ["job_description", "daily_tasks", "work_snippet"],
        "Airbnb": ["name", "amenities"], "consumer_complaint": ["consumer_complaint_narrative"],
        "fake_job_postings": ["description"], "Kickstarter": ["name"],
        "netflix_titles": ["title", "cast", "description"],
    }
    gap_rows = []
    for ds in ds_order:
        gaps5, gaps0 = [], []
        for col in cols_by_ds[ds]:
            col_results = all_results.get(ds, {}).get(col, {})
            if K0_BASE_MODEL in col_results and K0_MODEL in col_results:
                real = col_results[K0_BASE_MODEL]["wc_mean_real"]
                gaps5.append(abs(col_results[K0_BASE_MODEL]["wc_mean_synthetic"] - real))
                gaps0.append(abs(col_results[K0_MODEL]["wc_mean_synthetic"] - real))
        if not gaps5:
            continue
        gap5m, gap0m = float(np.mean(gaps5)), float(np.mean(gaps0))
        gap_rows.append(f"{_DS_MACRO[ds]:12} & {_fmt(gap5m,1)} & {_fmt(gap0m,1)} \\\\")
    body = "\n".join(gap_rows)
    return r"""\begin{table}[htbp]
\centering
\small
\caption{\textbf{Retrieval ablation: word count.} Unsigned gap between mean synthetic word count and the real reference value, averaged across each dataset's free-text columns, for the \Gfl generator with ($k=5$) and without ($k=0$) retrieval context. Lower is closer to real. Real reference values are reported in Table~\ref{tab:linguistic-fidelity}. MATTR and Distinct-2 showed no consistent direction between $k=5$ and $k=0$ across datasets and are omitted here; see Table~\ref{tab:linguistic-fidelity} for their $k=5$ values.}
\label{tab:k0-linguistic}
\begin{tabular}{lcc}
\toprule
\textbf{Dataset} & \textbf{Gap ($k=5$)} & \textbf{Gap ($k=0$)} \\ \midrule
""" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


_GEN_LABEL = {"gemini-3.1-flash-lite-preview": r"\Gfl", "gpt-4o-mini": r"\GPT",
              "Qwen/Qwen2.5-14B-Instruct": r"\Qfourteen"}


def build_combined_generator_table(all_results: dict[str, Any]) -> str:
    """Single compact appendix table with per-generator synthetic values only
    (no Real columns -- Real is already reported once in Table~tab:linguistic-fidelity)."""
    rows = []
    for ds, col in _COL_ORDER:
        col_results = all_results.get(ds, {}).get(col, {})
        if not col_results:
            continue
        col_disp = col.replace("consumer_complaint_narrative", "narrative").replace("_", chr(92) + "_")
        label = f"{_DS_MACRO[ds]} ({col_disp})"
        cells = []
        for field, nd in (("wc_mean_synthetic", 1), ("mattr_synthetic", 3), ("distinct2_synthetic", 3)):
            for model in MAIN_MODELS:
                v = col_results.get(model, {}).get(field, float("nan"))
                cells.append(_fmt(v, nd))
        rows.append(f"{label:32} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    return r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{\textbf{Free-text linguistic fidelity by generator.} Synthetic values for \Gfl, \GPT, and \Qfourteen individually; real-value references are reported once in Table~\ref{tab:linguistic-fidelity} and omitted here for space. \GPT's \dsComplaint value uses a separate $300$-row sample, as noted in Table~\ref{tab:linguistic-fidelity}.}
\label{tab:linguistic-fidelity-bygen}
\begin{tabular}{lccc ccc ccc}
\toprule
 & \multicolumn{3}{c}{Words} & \multicolumn{3}{c}{MATTR} & \multicolumn{3}{c}{Distinct-2} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
\textbf{Dataset (column)} & \Gfl & \GPT & \Qfourteen & \Gfl & \GPT & \Qfourteen & \Gfl & \GPT & \Qfourteen \\ \midrule
""" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def write_latex(all_results: dict[str, Any]) -> None:
    main_table = build_main_table(all_results)
    ablation_table = build_ablation_table(all_results)
    per_gen_table = build_combined_generator_table(all_results)

    methods_para = r"""\subsection{Linguistic fidelity metrics}
\label{app:linguistic-fidelity}

For each free-text column, we compare surface-level linguistic properties of the synthetic text with those of the real data, motivated by evidence that LLM-generated text is less lexically and syntactically diverse than human text~\citep{guo2025linguisticdiversity,shaib2024templates}. We report the mean document length (whitespace word count), lexical diversity using the moving-average type--token ratio (MATTR)~\citep{covington2010mattr} with a window of 100 lowercased word tokens (a separate, regex-based tokenizer from the whitespace word-count above, chosen to avoid inflating apparent diversity on punctuation- or list-like fields such as \dsAirbnb amenities), and phrase repetition with corpus-level Distinct-2, the fraction of unique within-document bigrams. Unlike the structured-data fidelity metrics, free-text generation does not fit a model to a training split -- retrieval only reads individual real examples at generation time -- so we report real-holdout and synthetic values directly rather than using a train-vs-holdout calibration baseline.

Because type--token statistics depend on text length, both lexical metrics are computed at a fixed token budget equal to the smallest token count among all corpora compared for that column (the real holdout data and every synthetic variant, including the $k=0$ ablation). Each corpus is sampled to this budget by shuffling whole documents, and results are averaged over five random seeds.
"""

    results_para = r"""\subsection{Linguistic fidelity results}
\label{app:linguistic-fidelity-results}

Table~\ref{tab:linguistic-fidelity} reports the three linguistic fidelity metrics across the three $k=5$ NIFTSy configurations, and Table~\ref{tab:k0-linguistic} compares word count with and without retrieval for the \Gfl generator; MATTR and Distinct-2 showed no consistent direction between retrieval settings across datasets and are omitted from that comparison. Table~\ref{tab:linguistic-fidelity-bygen} reports the complete per-generator results underlying Table~\ref{tab:linguistic-fidelity}'s averages.
"""

    out = "% AUTO-GENERATED by source/metrics/run_linguistic_fidelity.py -- not included from main.tex.\n"
    out += "% Paste the relevant pieces into main.tex (near tab:full-quality and tab:k0-ablation,\n"
    out += "% and into the evaluation-details / k0-ablation appendices) once reviewed.\n\n"
    out += "\\paragraph{Linguistic fidelity.}\n% TODO: one lead-in sentence, see app:linguistic-fidelity-results for a starting point.\n\n"
    out += main_table + "\n" + ablation_table + "\n" + methods_para + "\n" + results_para
    out += "\n" + per_gen_table

    TEX_OUTPUT_PATH.write_text(out)
    print(f"Saved: {TEX_OUTPUT_PATH}")


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

    write_latex(all_results)


if __name__ == "__main__":
    main()
