from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from source.metrics.structured_metrics import (
    evaluate_joint_holdout_discriminator,
    evaluate_structured_qa_metrics,
    evaluate_tstr,
)
from source.metrics.free_text_metrics import evaluate_first_free_text_distribution


DEFAULT_CONFIG = Path("configs/metrics.yml")
DEFAULT_METADATA_ROOT = Path(".metadata")


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(data)}")
    return data


def _safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-") or "model"


def _resolve_path(template: str, *, dataset_name: str, model: str) -> Path:
    safe_gen = _safe_model_name(model)
    return Path(template.format(dataset_name=dataset_name, model=safe_gen, generator_model=safe_gen))


def _dataset_name_from_path(input_csv: Path) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", input_csv.stem).strip("-")
    return sanitized or "dataset"


def _metadata_path(dataset_name: str) -> Path:
    return DEFAULT_METADATA_ROOT / dataset_name / "metadata.json"


def _resolve_json_metadata_path(dataset_name: str, configured_json_path: Any | None) -> Path:
    if configured_json_path is None:
        return _metadata_path(dataset_name)

    configured = str(configured_json_path).strip()
    if configured.lower() in {"", "none", "null"}:
        return _metadata_path(dataset_name)

    resolved = configured.format(dataset_name=dataset_name)
    return Path(resolved)


def _load_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata file must be a JSON object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structured-column QA metrics using real and synthetic datasets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset identifier to override dataset_name from metrics.yml.",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default=None,
        help="Override generator_model from metrics.yml.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20_000,
        help="Cap real and synthetic data at this many rows (random sample) before "
        "computing any metric, so results are comparable ('apples to apples') across "
        "datasets of very different sizes. Pass 0 to disable capping.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = _read_yaml(args.config)

    dataset_name = args.dataset_name or cfg.get("dataset_name")
    model = args.generator_model or cfg.get("generator_model") or cfg.get("model")
    real_template = cfg.get("real_data_path")
    synthetic_template = cfg.get("synthetic_data_path")
    output_template = cfg.get("report_output_path")

    if not isinstance(model, str) or not model.strip():
        raise ValueError("metrics.yml must include a non-empty string key: generator_model")

    if (
        not isinstance(real_template, str)
        or not isinstance(synthetic_template, str)
        or not isinstance(output_template, str)
    ):
        raise ValueError(
            "metrics.yml must include string keys: real_data_path, synthetic_data_path, and report_output_path."
        )

    holdout_fraction = float(cfg.get("holdout_fraction", 0.2))
    random_seed = int(cfg.get("random_state", 42))
    text_embedding_dim = int(cfg.get("text_embedding_dim", 768))
    text_max_features = int(cfg.get("text_max_features", 200_000))
    text_mmd_batch_size = int(cfg.get("text_mmd_batch_size", 4096))
    tstr_structured_weight = float(cfg.get("tstr_structured_weight", 1.0))
    tstr_text_weight_default = float(cfg.get("tstr_text_weight_default", 0.01))
    tstr_text_weight_by_column = cfg.get("tstr_text_weight_by_column", {})
    if not isinstance(tstr_text_weight_by_column, dict):
        raise ValueError("tstr_text_weight_by_column must be a mapping in metrics.yml.")

    if isinstance(dataset_name, str) and dataset_name.strip():
        dataset_name = dataset_name.strip()
    else:
        if not isinstance(real_template, str):
            raise ValueError(
                "Either set dataset_name in metrics.yml or provide a string real_data_path to infer it."
            )
        dataset_name = _dataset_name_from_path(Path(real_template))

    metadata_json_path = cfg.get("metadata_json_path")
    metadata_path = _resolve_json_metadata_path(dataset_name, metadata_json_path)
    metadata = _load_metadata(metadata_path)

    if metadata is None and isinstance(real_template, str):
        inferred_dataset_name = _dataset_name_from_path(Path(real_template))
        if inferred_dataset_name != dataset_name:
            inferred_metadata_path = _resolve_json_metadata_path(
                inferred_dataset_name,
                metadata_json_path,
            )
            inferred_metadata = _load_metadata(inferred_metadata_path)
            if inferred_metadata is not None:
                dataset_name = inferred_dataset_name
                metadata_path = inferred_metadata_path
                metadata = inferred_metadata

    config_text_columns = cfg.get("free_text_columns", [])
    if not isinstance(config_text_columns, list):
        raise ValueError("free_text_columns must be a list in metrics.yml")

    if metadata is not None:
        metadata_dataset_name = metadata.get("dataset_name")
        if isinstance(metadata_dataset_name, str) and metadata_dataset_name.strip():
            dataset_name = metadata_dataset_name.strip()

        metadata_text_columns = metadata.get("free_text_columns") or []
        if not isinstance(metadata_text_columns, list):
            raise ValueError("'free_text_columns' in metadata.json must be a list.")
        text_columns = [str(col) for col in metadata_text_columns]
        metadata_target_column = metadata.get("target_column")
        target_column = str(metadata_target_column).strip() if metadata_target_column else None
        print(f"Using metadata from: {metadata_path}")
    else:
        text_columns = [str(col) for col in config_text_columns]
        cfg_target_column = cfg.get("target_column")
        target_column = str(cfg_target_column).strip() if cfg_target_column else None
        print(f"Metadata JSON not found at {metadata_path}; using free_text_columns from config.")

    real_path = _resolve_path(real_template, dataset_name=dataset_name, model=model)
    synthetic_path = _resolve_path(synthetic_template, dataset_name=dataset_name, model=model)
    output_path = _resolve_path(output_template, dataset_name=dataset_name, model=model)

    if not real_path.exists():
        raise FileNotFoundError(f"Real data not found at: {real_path}")
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Synthetic data not found at: {synthetic_path}")

    real_df = pd.read_csv(real_path)
    synthetic_df = pd.read_csv(synthetic_path)

    max_rows = args.max_rows
    if max_rows and max_rows > 0:
        if len(real_df) > max_rows:
            print(f"Sampling real data down to {max_rows} rows (from {len(real_df)}).")
            real_df = real_df.sample(max_rows, random_state=random_seed).reset_index(drop=True)
        if len(synthetic_df) > max_rows:
            print(f"Sampling synthetic data down to {max_rows} rows (from {len(synthetic_df)}).")
            synthetic_df = synthetic_df.sample(max_rows, random_state=random_seed).reset_index(drop=True)

    if target_column and target_column not in real_df.columns:
        print(f"Configured target_column '{target_column}' does not exist in real data; skipping TSTR.")
        target_column = None

    structured_columns = [col for col in real_df.columns if col not in set(text_columns)]
    if not structured_columns:
        raise ValueError("No structured columns remained after removing free_text_columns.")

    print(f"Evaluating structured QA metrics on columns: {structured_columns}")
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
        n_repeats=int(cfg.get("qa_repeats", 3)),
    )

    quality_report_data: dict[str, Any]
    try:
        from sdv.evaluation.single_table import evaluate_quality
        from sdv.metadata import SingleTableMetadata

        metadata_sdv = SingleTableMetadata()
        metadata_sdv.detect_from_dataframe(real_df[structured_columns])
        quality_report = evaluate_quality(
            real_data=real_df[structured_columns],
            synthetic_data=synthetic_df[structured_columns],
            metadata=metadata_sdv,
        )
        quality_report_data = {
            "overall_score": float(quality_report.get_score()),
            "properties": quality_report.get_properties().to_dict("records"),
        }
    except Exception as exc:  # pragma: no cover - environment-specific dependency issues
        quality_report_data = {
            "error": str(exc),
        }
        print(f"Warning: unable to compute SDV quality report: {exc}")
    
    print(f"Evaluating free-text distribution metrics on columns: {text_columns} (if any)")
    free_text_distribution: dict[str, Any] | None = None
    if text_columns:
        free_text_distribution = {}
        for column_name in text_columns:
            print(f"[free-text] Processing column: {column_name}")
            try:
                free_text_distribution[column_name] = evaluate_first_free_text_distribution(
                    real_df=real_train,
                    synthetic_df=synthetic_df,
                    real_holdout_df=real_holdout,
                    free_text_columns=[column_name],
                    embedding_dim=text_embedding_dim,
                    max_features=text_max_features,
                    mmd_batch_size=text_mmd_batch_size,
                    show_batch_progress=True,
                    progress_prefix=f"{column_name}",
                )
                print(f"[free-text] Finished metrics for column: {column_name}")
            except Exception as exc:  # pragma: no cover - dataset/config dependent
                free_text_distribution[column_name] = {"error": str(exc)}
                print(f"Warning: unable to compute free-text distribution metrics for '{column_name}': {exc}")

    joint_holdout_discriminator: dict[str, Any] | None = None
    if structured_columns or text_columns:
        print("Evaluating joint structured + free-text holdout discriminator...")
        try:
            joint_holdout_discriminator = evaluate_joint_holdout_discriminator(
                real_train_df=real_train.reset_index(drop=True),
                real_holdout_df=real_holdout.reset_index(drop=True),
                synthetic_df=synthetic_df.reset_index(drop=True),
                text_columns=text_columns,
                structured_weight=tstr_structured_weight,
                text_weight_default=tstr_text_weight_default,
                text_weight_by_column={
                    str(key): float(value) for key, value in tstr_text_weight_by_column.items()
                },
                text_embedding_dim=text_embedding_dim,
                text_max_features=text_max_features,
                random_state=random_seed,
            )
        except Exception as exc:  # pragma: no cover - dataset/config dependent
            joint_holdout_discriminator = {"error": str(exc)}
            print(f"Warning: unable to compute joint holdout discriminator metrics: {exc}")

    tstr_data: dict[str, Any] | None = None
    if target_column:
        print(f"Evaluating TSTR metrics using target column: {target_column}")
        try:
            tstr_data = evaluate_tstr(
                train_df=real_train,
                synthetic_df=synthetic_df,
                holdout_df=real_holdout,
                target_column=target_column,
                text_columns=text_columns,
                structured_weight=tstr_structured_weight,
                text_weight_default=tstr_text_weight_default,
                text_weight_by_column={
                    str(key): float(value) for key, value in tstr_text_weight_by_column.items()
                },
                text_embedding_dim=text_embedding_dim,
                text_max_features=text_max_features,
            )
        except Exception as exc:  # pragma: no cover - dataset/config dependent
            tstr_data = {"error": str(exc), "target_column": target_column}
            print(f"Warning: unable to compute TSTR metrics: {exc}")
    else:
        tstr_data = {"skipped": True, "reason": "No valid target_column was configured."}
        print("Skipping TSTR metrics because no valid target_column was configured.")

    payload = {
        "dataset_name": dataset_name,
        "model": model,
        "real_data_path": str(real_path),
        "synthetic_data_path": str(synthetic_path),
        "structured_columns": structured_columns,
        "free_text_columns": text_columns,
        "metrics_qa": metrics,
        "free_text_distribution": free_text_distribution,
        "quality_report": quality_report_data,
        "joint_holdout_discriminator": joint_holdout_discriminator,
        "tstr": tstr_data,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved structured QA metrics to: {output_path}")


if __name__ == "__main__":
    main()
