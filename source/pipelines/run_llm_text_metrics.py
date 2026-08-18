from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors


DEFAULT_CONFIG = Path("configs/metrics.yml")
DEFAULT_METADATA_ROOT = Path(".metadata")
OUTPUT_DIR = Path("reports/llm_metrics")


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(data)}")
    return data


def _safe_model_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "model"


def _load_metadata(dataset_name: str) -> dict[str, Any] | None:
    path = DEFAULT_METADATA_ROOT / dataset_name / "metadata.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else None


def _encode_structured(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Min-max encode structured columns to [0,1] for distance computation."""
    parts: list[np.ndarray] = []
    for col in columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            vals = series.fillna(series.median()).values.astype(float).reshape(-1, 1)
        else:
            le = LabelEncoder()
            vals = le.fit_transform(series.fillna("__NA__").astype(str)).reshape(-1, 1).astype(float)
        mn, mx = vals.min(), vals.max()
        if mx > mn:
            vals = (vals - mn) / (mx - mn)
        parts.append(vals)
    return np.hstack(parts) if parts else np.zeros((len(df), 1))


def evaluate_bertscore(
    real_holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    free_text_columns: list[str],
    *,
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 64,
    device: str = "cuda",
    rng: np.random.Generator,
) -> dict[str, Any]:
    from bert_score import score as bert_score_fn

    results: dict[str, Any] = {}
    for col in free_text_columns:
        if col not in real_holdout_df.columns or col not in synthetic_df.columns:
            results[col] = {"error": f"column '{col}' not found"}
            continue
        try:
            synt_texts = synthetic_df[col].dropna().astype(str).tolist()
            real_texts = real_holdout_df[col].dropna().astype(str).tolist()

            # BERTScore needs paired lists — sample holdout with replacement to match synthetic size
            idx = rng.integers(0, len(real_texts), size=len(synt_texts))
            ref_texts = [real_texts[i] for i in idx]

            P, R, F1 = bert_score_fn(
                synt_texts,
                ref_texts,
                lang="en",
                model_type=model_type,
                batch_size=batch_size,
                device=device,
                verbose=False,
            )
            results[col] = {
                "bertscore_precision_mean": float(P.mean()),
                "bertscore_recall_mean": float(R.mean()),
                "bertscore_f1_mean": float(F1.mean()),
                "bertscore_f1_std": float(F1.std()),
                "bertscore_model": model_type,
                "n_synthetic": len(synt_texts),
                "n_reference_pool": len(real_texts),
            }
            print(f"  [{col}] BERTScore F1: {float(F1.mean()):.4f} ± {float(F1.std()):.4f}")
        except Exception as exc:
            results[col] = {"error": str(exc)}
            print(f"Warning: BERTScore failed for '{col}': {exc}")
    return results


def evaluate_mauve(
    real_holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    free_text_columns: list[str],
    *,
    max_samples: int = 500,
    device_id: int = 0,
    rng: np.random.Generator,
) -> dict[str, Any]:
    import mauve as mauve_lib

    results: dict[str, Any] = {}
    for col in free_text_columns:
        if col not in real_holdout_df.columns or col not in synthetic_df.columns:
            results[col] = {"error": f"column '{col}' not found"}
            continue
        try:
            synt_texts = synthetic_df[col].dropna().astype(str).tolist()
            real_texts = real_holdout_df[col].dropna().astype(str).tolist()

            if len(synt_texts) > max_samples:
                synt_texts = [synt_texts[i] for i in rng.choice(len(synt_texts), max_samples, replace=False)]
            if len(real_texts) > max_samples:
                real_texts = [real_texts[i] for i in rng.choice(len(real_texts), max_samples, replace=False)]

            out = mauve_lib.compute_mauve(
                p_text=synt_texts,
                q_text=real_texts,
                device_id=device_id,
                max_text_length=512,
                verbose=False,
            )
            results[col] = {
                "mauve_score": float(out.mauve),
                "n_synthetic_sampled": len(synt_texts),
                "n_real_sampled": len(real_texts),
            }
            print(f"  [{col}] MAUVE: {float(out.mauve):.4f}")
        except Exception as exc:
            results[col] = {"error": str(exc)}
            print(f"Warning: MAUVE failed for '{col}': {exc}")
    return results


def evaluate_privacy_nndr(
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    structured_columns: list[str],
    *,
    nndr_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Nearest-Neighbor Distance Ratio (NNDR) privacy proxy.

    For each query row:  nndr = dist_to_nearest_real_train / dist_to_2nd_nearest_self_set

    Lower NNDR for synthetic → rows are suspiciously close to real training records (memorisation risk).
    Calibrated against holdout so scores are interpretable: if synthetic NNDR ≈ holdout NNDR the
    synthetic is no closer to training data than genuinely unseen real examples.
    """
    usable = [c for c in structured_columns if c in real_train_df.columns and c in synthetic_df.columns]
    if not usable:
        return {"error": "no overlapping structured columns for NNDR"}

    try:
        train_enc = _encode_structured(real_train_df, usable)
        holdout_enc = _encode_structured(real_holdout_df, usable)
        synth_enc = _encode_structured(synthetic_df, usable)

        knn_train = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1).fit(train_enc)

        def _nndr(query: np.ndarray, self_set: np.ndarray) -> np.ndarray:
            d_real, _ = knn_train.kneighbors(query)
            d_real = d_real[:, 0]
            knn_self = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=-1).fit(self_set)
            d_self, _ = knn_self.kneighbors(query)
            d_self_nn = d_self[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(d_self_nn > 0, d_real / d_self_nn, np.inf)

        nndr_synth = _nndr(synth_enc, synth_enc)
        nndr_holdout = _nndr(holdout_enc, holdout_enc)

        finite_s = nndr_synth[np.isfinite(nndr_synth)]
        finite_h = nndr_holdout[np.isfinite(nndr_holdout)]

        result = {
            "nndr_synthetic_mean": float(np.mean(finite_s)),
            "nndr_synthetic_median": float(np.median(finite_s)),
            "nndr_synthetic_share_below_threshold": float(np.mean(nndr_synth < nndr_threshold)),
            "nndr_holdout_mean": float(np.mean(finite_h)),
            "nndr_holdout_median": float(np.median(finite_h)),
            "nndr_holdout_share_below_threshold": float(np.mean(nndr_holdout < nndr_threshold)),
            "nndr_threshold": nndr_threshold,
            "n_structured_columns_used": len(usable),
        }
        print(
            f"  NNDR  synthetic mean={result['nndr_synthetic_mean']:.4f}"
            f"  holdout mean={result['nndr_holdout_mean']:.4f}"
            f"  (threshold={nndr_threshold})"
        )
        return result
    except Exception as exc:
        return {"error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BERTScore, MAUVE, and privacy NNDR for synthetic text data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--generator-model", type=str, default=None,
                        help="Override generator_model from metrics.yml.")
    parser.add_argument("--mauve-max-samples", type=int, default=500)
    parser.add_argument("--bertscore-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--nndr-threshold", type=float, default=0.5)
    parser.add_argument("--max-synthetic-rows", type=int, default=None,
                        help="Randomly sample the synthetic data down to this many rows "
                        "before computing BERTScore/MAUVE (unbounded by default). "
                        "BERTScore has no internal cap and scales with the full synthetic "
                        "row count, so large datasets (e.g. Airbnb at ~280k rows) can take "
                        "a very long time without this.")
    parser.add_argument("--device-id", type=int, default=0,
                        help="CUDA device id for MAUVE (BERTScore uses cuda:0 within CUDA_VISIBLE_DEVICES).")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = _read_yaml(args.config)

    dataset_name = args.dataset_name or cfg.get("dataset_name")
    model = args.generator_model or cfg.get("generator_model") or cfg.get("model")
    real_template = cfg.get("real_data_path", "data/raw/{dataset_name}.csv")
    synthetic_template = cfg.get("synthetic_data_path", "data/processed/{dataset_name}_{generator_model}.csv")

    if not isinstance(model, str) or not model.strip():
        raise ValueError("Provide generator_model in metrics.yml or via --generator-model")
    if not dataset_name:
        raise ValueError("Provide dataset_name in metrics.yml or via --dataset-name")

    holdout_fraction = float(cfg.get("holdout_fraction", 0.2))
    random_seed = int(cfg.get("random_state", 42))

    safe_model = _safe_model_name(model)
    real_path = Path(real_template.format(dataset_name=dataset_name, generator_model=safe_model))
    synthetic_path = Path(synthetic_template.format(dataset_name=dataset_name, generator_model=safe_model))
    output_path = OUTPUT_DIR / f"{dataset_name}_{safe_model}_llm_text_metrics.json"

    if not real_path.exists():
        raise FileNotFoundError(f"Real data not found: {real_path}")
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Synthetic data not found: {synthetic_path}")

    # Resolve free-text columns from metadata
    metadata = _load_metadata(dataset_name)
    if metadata is not None:
        text_columns = [str(c) for c in (metadata.get("free_text_columns") or [])]
        print(f"Loaded free_text_columns from metadata: {text_columns}")
    else:
        text_columns = []
        print(f"No metadata found for '{dataset_name}'; no free-text columns.")

    real_df = pd.read_csv(real_path)
    synthetic_df = pd.read_csv(synthetic_path)
    structured_columns = [c for c in real_df.columns if c not in set(text_columns)]

    if args.max_synthetic_rows is not None and len(synthetic_df) > args.max_synthetic_rows:
        print(f"Sampling synthetic data down to {args.max_synthetic_rows} rows "
              f"(from {len(synthetic_df)}) for BERTScore/MAUVE.")
        synthetic_df = synthetic_df.sample(args.max_synthetic_rows, random_state=random_seed).reset_index(drop=True)

    real_train, real_holdout = train_test_split(real_df, test_size=holdout_fraction, random_state=random_seed)
    rng = np.random.default_rng(random_seed)

    payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "model": model,
        "real_data_path": str(real_path),
        "synthetic_data_path": str(synthetic_path),
        "free_text_columns": text_columns,
        "structured_columns": structured_columns,
    }

    if text_columns:
        print(f"\nComputing BERTScore for: {text_columns}")
        payload["bertscore"] = evaluate_bertscore(
            real_holdout, synthetic_df, text_columns,
            model_type=args.bertscore_model,
            rng=rng,
        )

        print(f"\nComputing MAUVE for: {text_columns} (max {args.mauve_max_samples} samples)")
        payload["mauve"] = evaluate_mauve(
            real_holdout, synthetic_df, text_columns,
            max_samples=args.mauve_max_samples,
            device_id=args.device_id,
            rng=rng,
        )
    else:
        payload["bertscore"] = {"skipped": True, "reason": "no free_text_columns"}
        payload["mauve"] = {"skipped": True, "reason": "no free_text_columns"}

    print(f"\nComputing privacy NNDR on {len(structured_columns)} structured columns...")
    payload["privacy_nndr"] = evaluate_privacy_nndr(
        real_train, real_holdout, synthetic_df, structured_columns,
        nndr_threshold=args.nndr_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
