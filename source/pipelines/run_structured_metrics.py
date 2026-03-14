from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from source.metrics.structured_metrics import evaluate_structured_qa_metrics


DEFAULT_CONFIG = Path("configs/metrics.yml")


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(data)}")
    return data


def _safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-") or "model"


def _resolve_path(template: str, *, dataset_name: str, model: str) -> Path:
    return Path(template.format(dataset_name=dataset_name, model=_safe_model_name(model)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structured-column QA metrics using real and synthetic datasets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _read_yaml(args.config)

    real_template = cfg.get("real_data_path")
    synthetic_template = cfg.get("synthetic_data_path")
    output_template = cfg.get("report_output_path")

    if not isinstance(real_template, str) or not isinstance(synthetic_template, str) or not isinstance(output_template, str):
        raise ValueError(
            "metrics.yml must include string keys: real_data_path, synthetic_data_path, and report_output_path."
        )

    text_columns = cfg.get("free_text_columns", [])
    if not isinstance(text_columns, list):
        raise ValueError("free_text_columns must be a list in metrics.yml")

    holdout_fraction = float(cfg.get("holdout_fraction", 0.2))
    random_seed = int(cfg.get("random_state", 42))

    real_path = _resolve_path(real_template, dataset_name=args.dataset_name, model=args.model)
    synthetic_path = _resolve_path(synthetic_template, dataset_name=args.dataset_name, model=args.model)
    output_path = _resolve_path(output_template, dataset_name=args.dataset_name, model=args.model)

    if not real_path.exists():
        raise FileNotFoundError(f"Real data not found at: {real_path}")
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Synthetic data not found at: {synthetic_path}")

    real_df = pd.read_csv(real_path)
    synthetic_df = pd.read_csv(synthetic_path)

    structured_columns = [col for col in real_df.columns if col not in set(text_columns)]
    if not structured_columns:
        raise ValueError("No structured columns remained after removing free_text_columns.")

    real_train, real_holdout = train_test_split(
        real_df,
        test_size=holdout_fraction,
        random_state=random_seed,
    )

    metrics = evaluate_structured_qa_metrics(
        train_df=real_train,
        synthetic_df=synthetic_df,
        holdout_df=real_holdout,
        non_free_text_columns=structured_columns,
    )

    payload = {
        "dataset_name": args.dataset_name,
        "model": args.model,
        "real_data_path": str(real_path),
        "synthetic_data_path": str(synthetic_path),
        "structured_columns": structured_columns,
        "metrics_qa": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved structured QA metrics to: {output_path}")


if __name__ == "__main__":
    main()
