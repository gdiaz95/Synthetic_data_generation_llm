from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

DEFAULT_CONFIG = Path("configs/metrics.yml")
OUTPUT_DIR = Path("reports/membership")


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_path(template: str, *, dataset_name: str, model: str) -> Path:
    return Path(template.format(dataset_name=dataset_name, generator_model=model.replace("/", "-")))


def _encode(train_df: pd.DataFrame, holdout_df: pd.DataFrame, syn_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = pd.concat([train_df, holdout_df, syn_df], ignore_index=True)
    parts: list[np.ndarray] = []
    for col in combined.columns:
        col_data = combined[col]
        if pd.api.types.is_numeric_dtype(col_data):
            vals = col_data.fillna(col_data.mean()).to_numpy(float)
            std = vals.std()
            parts.append(((vals - vals.mean()) / std if std > 0 else vals).reshape(-1, 1))
        else:
            codes, _ = pd.factorize(col_data.fillna("__NA__").astype(str))
            parts.append(codes.astype(float).reshape(-1, 1))
    X = np.hstack(parts) if parts else np.zeros((len(combined), 1))
    n_tr, n_ho = len(train_df), len(holdout_df)
    return X[:n_tr], X[n_tr:n_tr + n_ho], X[n_tr + n_ho:]


def run_mia(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    structured_columns: list[str] | None = None,
) -> dict:
    cols = structured_columns or list(real_train.columns)
    cols = [c for c in cols if c in real_train.columns and c in synthetic_df.columns]

    X_train, X_holdout, X_syn = _encode(
        real_train[cols], real_holdout[cols], synthetic_df[cols]
    )
    X_real = np.vstack([X_train, X_holdout])
    labels = np.array([1] * len(X_train) + [0] * len(X_holdout))

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=1)
    nn.fit(X_syn)
    dists = nn.kneighbors(X_real)[0][:, 0]

    auroc = float(roc_auc_score(labels, -dists))
    return {
        "mia_auroc": round(auroc, 4),
        "n_train": int(len(real_train)),
        "n_holdout": int(len(real_holdout)),
        "n_synthetic": int(len(synthetic_df)),
        "mean_dist_train": round(float(dists[labels == 1].mean()), 4),
        "mean_dist_holdout": round(float(dists[labels == 0].mean()), 4),
        "columns_used": cols,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distance-based membership inference attack on synthetic data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--generator-model", type=str, default="gemini-3.1-flash-lite-preview")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = _read_yaml(args.config)

    dataset_name = args.dataset_name or cfg.get("dataset_name")
    model = args.generator_model or cfg.get("generator_model") or cfg.get("model")
    holdout_fraction = float(cfg.get("holdout_fraction", 0.2))
    random_state = int(cfg.get("random_state", 42))

    real_path = _resolve_path(cfg["real_data_path"], dataset_name=dataset_name, model=model)
    synthetic_path = _resolve_path(cfg["synthetic_data_path"], dataset_name=dataset_name, model=model)

    print(f"Dataset : {dataset_name}")
    print(f"Model   : {model}")
    print(f"Real    : {real_path}")
    print(f"Synthetic: {synthetic_path}")

    real_df = pd.read_csv(real_path)
    synthetic_df = pd.read_csv(synthetic_path)

    real_train, real_holdout = train_test_split(
        real_df, test_size=holdout_fraction, random_state=random_state
    )

    structured_columns: list[str] | None = None
    meta_path = Path(f".metadata/{dataset_name}/metadata.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        free_text = meta.get("text_columns") or meta.get("free_text_columns") or []
        if free_text:
            structured_columns = [c for c in real_df.columns if c not in free_text]
            print(f"Excluding free-text columns: {free_text}")

    result = run_mia(real_train, real_holdout, synthetic_df, structured_columns)

    print(f"\nMIA AUROC : {result['mia_auroc']:.4f}  (0.5 = no leakage, 1.0 = full memorization)")
    print(f"Mean dist train   : {result['mean_dist_train']:.4f}")
    print(f"Mean dist holdout : {result['mean_dist_holdout']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset_name}_membership.json"
    out_path.write_text(json.dumps({"dataset": dataset_name, **result}, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
