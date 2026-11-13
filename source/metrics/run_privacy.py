"""Privacy metrics (identical-match rate, DCR share, MIA) on NPGC fit on the
training split only.

The privacy metrics compare how close real training rows and real holdout rows
are to the synthetic data, so the holdout must be unseen by the synthesizer.
The privacy metrics use only the structured columns, which NIFTSy generates
with NPGC alone (the LLM never changes them), so each run fits NPGC on the
training split and samples a structured synthetic dataset; no LLM is involved.

The real data is capped and split exactly as in the other metric scripts
(real data capped at 20,000 rows with random_state, then train_test_split with
holdout_fraction and random_state). NPGC is fit and sampled exactly as in
run_full.py (configs/tabular.yml settings, unseeded), and the synthetic set has
min(real rows, 20,000) rows, as in NIFTSy. Each run (--run 1, 2, 3) fits and
samples NPGC once; the paper reports the mean over the three runs. The MIA is
reported on all structured columns and, separately, without high-cardinality
label columns (see main()).

Usage:
    python -m source.metrics.run_privacy --dataset-name <name> --run <1|2|3>

Output: reports/privacy/{dataset}_run{run}_privacy.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from source.metrics.run_membership import run_mia
from source.metrics.structured_metrics import evaluate_structured_qa_metrics
from source.tabular.npgc import NPGC_special

METRICS_CONFIG = Path("configs/metrics.yml")
TABULAR_CONFIG = Path("configs/tabular.yml")
OUTPUT_DIR = Path("reports/privacy")
MAX_ROWS = 20_000
HIGH_CARDINALITY_SHARE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--run", type=int, required=True, help="Run index (1, 2 or 3).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(METRICS_CONFIG.read_text())
    tabular_cfg = yaml.safe_load(TABULAR_CONFIG.read_text())
    holdout_fraction = float(cfg.get("holdout_fraction", 0.2))
    random_state = int(cfg.get("random_state", 42))
    dataset_name = args.dataset_name

    real_path = Path(cfg["real_data_path"].format(dataset_name=dataset_name))
    real_df = pd.read_csv(real_path)
    if len(real_df) > MAX_ROWS:
        real_df = real_df.sample(MAX_ROWS, random_state=random_state).reset_index(drop=True)
    real_train, real_holdout = train_test_split(real_df, test_size=holdout_fraction, random_state=random_state)

    metadata = json.loads(Path(f".metadata/{dataset_name}/metadata.json").read_text())
    text_columns = set(metadata.get("free_text_columns") or [])
    structured_columns = [c for c in real_df.columns if c not in text_columns]

    print(f"Dataset   : {dataset_name} (run {args.run})")
    print(f"Real      : {real_path} -> train {len(real_train)}, holdout {len(real_holdout)}")
    print(f"Columns   : {structured_columns}")

    synthesizer = NPGC_special(
        enforce_min_max_values=bool(tabular_cfg["enforce_min_max_values"]),
        epsilon=tabular_cfg["epsilon"],
    )
    synthesizer.fit(real_train[structured_columns])
    synthetic_df, _ = synthesizer.sample(len(real_df))
    print(f"Synthetic : {len(synthetic_df)} rows from NPGC fit on the training split")

    qa_metrics = evaluate_structured_qa_metrics(
        train_df=real_train,
        synthetic_df=synthetic_df,
        holdout_df=real_holdout,
        non_free_text_columns=structured_columns,
        n_repeats=int(cfg.get("qa_repeats", 3)),
    )
    mia = run_mia(real_train, real_holdout, synthetic_df, structured_columns)
    # NPGC samples only category labels seen in training, so name-like columns
    # (non-numeric, unique values in more than 5% of rows, e.g. titles or track
    # names) reveal which labels were in the training set. The second MIA
    # excludes these label columns to separate that effect from the rest.
    label_columns = [
        c for c in structured_columns
        if not pd.api.types.is_numeric_dtype(real_df[c]) and real_df[c].nunique() > HIGH_CARDINALITY_SHARE * len(real_df)
    ]
    other_columns = [c for c in structured_columns if c not in label_columns]
    mia_without_labels = run_mia(real_train, real_holdout, synthetic_df, other_columns) if other_columns else None

    result = {
        "dataset": dataset_name,
        "run": args.run,
        "n_train": int(len(real_train)),
        "n_holdout": int(len(real_holdout)),
        "n_synthetic": int(len(synthetic_df)),
        "identical_matches": qa_metrics["identical_matches"],
        "dcr_share": qa_metrics["dcr_share"],
        "qa_repeats": qa_metrics["n_repeats"],
        "qa_runs": qa_metrics["runs"],
        "mia": mia,
        "high_cardinality_label_columns": label_columns,
        "mia_without_label_columns": mia_without_labels,
    }
    print(f"Identical matches : {result['identical_matches']}")
    print(f"DCR share         : {result['dcr_share']:.4f}")
    print(f"MIA worst-case    : {mia['mia_auroc_worst_case']:.4f}")
    if mia_without_labels:
        print(f"MIA without {label_columns}: {mia_without_labels['mia_auroc_worst_case']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset_name}_run{args.run}_privacy.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
