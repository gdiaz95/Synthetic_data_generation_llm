from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import re
from typing import Any
import pandas as pd
import yaml
from dotenv import load_dotenv
import sys

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from source.llm.server_model import SingleGPUServer
from source.pipelines.run_free_text import generate_free_text_column
from source.pipelines.run_tabular import knn_retrieval_with_text_blocks, run_tabular_generation
from source.pipelines.text_vector_prep import prepare_text_vector_features


DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_METADATA_ROOT = Path(".metadata")
DEFAULT_LOG_DIR = Path("logs")


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(data)}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic tabular dataset with LLM-generated free-text columns.",
    )
    parser.add_argument("--tabular-config", type=Path, default=DEFAULT_CONFIG_DIR / "tabular.yml")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_CONFIG_DIR / "llm_generation.yml")
    parser.add_argument("--prompts-config", type=Path, default=DEFAULT_CONFIG_DIR / "prompts.yml")

    parser.add_argument("--input-csv", type=Path, help="Override input CSV path")
    parser.add_argument("--output-csv", type=Path, help="Override output CSV path")
    parser.add_argument(
        "--feature-weights-json",
        type=Path,
        help="Optional JSON file mapping tabular feature names to distance weights.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset identifier used to discover metadata at .metadata/{dataset_name}/metadata.json.",
    )
    parser.add_argument(
        "--free-text-threshold",
        type=float,
        default=0.8,
        help="Unique-ratio threshold used to flag likely free-text columns in interactive mode.",
    )

    parser.add_argument("--model", type=str, help="Override generation model")
    parser.add_argument("--provider", choices=["auto", "local", "gemini", "openai"], help="Override backend provider")
    parser.add_argument("--gpu-memory-utilization", type=float, help="Override GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, help="Override max model length")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager execution in local vLLM backend (usually slower).",
    )

    return parser.parse_args()


def _merge_with_cli(config: dict, args: argparse.Namespace) -> dict:
    merged = dict(config)
    cli_overrides = {
        "model": args.model,
        "provider": args.provider,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _with_model_suffix(output_csv: Path, model_name: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-") or "model"
    return output_csv.with_name(f"{output_csv.stem}_{safe_model}{output_csv.suffix}")


def _load_feature_weights(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Feature-weights JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Feature-weights JSON must be an object mapping column names to numbers.")

    parsed: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("Feature-weights JSON keys must be column names (strings).")
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid weight for column '{key}': {value}") from exc
    return parsed


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


def _write_run_log(output_csv: Path, run_details: dict[str, Any]) -> Path:
    log_path = DEFAULT_LOG_DIR / f"{output_csv.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(run_details, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return log_path


def _detect_free_text_candidates(df: pd.DataFrame, threshold: float) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
            continue
        non_null = series.dropna().astype(str)
        if non_null.empty:
            continue
        unique_ratio = float(non_null.nunique(dropna=True)) / float(len(non_null))
        avg_words = float(non_null.str.split().str.len().mean())
        if unique_ratio >= threshold and avg_words >= 3.0:
            candidates.append(col)
    return candidates


def _prompt_free_text_columns(df: pd.DataFrame, threshold: float) -> list[str]:
    detected = _detect_free_text_candidates(df, threshold)
    print("\nNo metadata was found; launching interactive metadata checks.")
    print(f"Detected likely free-text columns with unique_ratio >= {threshold:.0%} and avg words >= 3:")
    if detected:
        print(f"- {', '.join(detected)}")
    else:
        print("- (none detected)")

    print("Enter comma-separated free-text columns to use (press Enter to accept detected list).")
    for col in df.columns:
        print(f"  - {col}")
    response = input("> ").strip()
    print("Available columns:")
    for col in df.columns:
        print(f"  - {col}")
    selected = detected if not response else [item.strip() for item in response.split(",") if item.strip()]
    invalid = [col for col in selected if col not in df.columns]
    if invalid:
        raise ValueError(f"Unknown columns provided during interactive metadata setup: {invalid}")
    return selected


def _prompt_feature_weights(df: pd.DataFrame, excluded_columns: set[str]) -> dict[str, float] | None:
    structured_cols = [c for c in df.columns if c not in excluded_columns]
    if not structured_cols:
        return None

    answer = input("Apply custom feature weights (different from 1.0)? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        return None

    default_raw = input("Default weight for all structured columns [1.0]: ").strip()
    default_weight = 1.0 if not default_raw else float(default_raw)
    if default_weight < 0:
        raise ValueError("Default weight must be non-negative.")

    weights = {col: default_weight for col in structured_cols}
    print("Provide overrides in format column=value, separated by commas (example: age=2.0,income=0.5).")
    overrides = input("Overrides (optional): ").strip()
    if overrides:
        for pair in [item.strip() for item in overrides.split(",") if item.strip()]:
            if "=" not in pair:
                raise ValueError(f"Invalid override '{pair}'. Expected format column=value.")
            key, value = pair.split("=", 1)
            key = key.strip()
            if key not in weights:
                raise ValueError(f"Column '{key}' is not a structured column and cannot be weighted.")
            weight = float(value.strip())
            if weight < 0:
                raise ValueError(f"Weight for '{key}' must be non-negative.")
            weights[key] = weight
    return weights


def _resolve_metadata(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    configured_text_column: str | None,
    threshold: float,
    cli_feature_weights_path: Path | None,
    configured_json_path: Any | None,
) -> tuple[list[str], dict[str, float] | None]:
    metadata_file = _resolve_json_metadata_path(dataset_name, configured_json_path)
    configured_json_raw = "" if configured_json_path is None else str(configured_json_path).strip().lower()
    allow_interactive_fallback = configured_json_raw in {"", "none", "null"}
    metadata = _load_metadata(metadata_file)

    if metadata is not None:
        print(f"Using metadata from: {metadata_file}")
        free_text_columns = metadata.get("free_text_columns") or []
        if not isinstance(free_text_columns, list):
            raise ValueError("'free_text_columns' in metadata.json must be a list.")
        text_columns = [str(col) for col in free_text_columns]

        metadata_weights = metadata.get("feature_weights")
        feature_weights: dict[str, float] | None = None
        if metadata_weights is not None:
            if not isinstance(metadata_weights, dict):
                raise ValueError("'feature_weights' in metadata.json must be an object.")
            feature_weights = {str(k): float(v) for k, v in metadata_weights.items()}
    else:
        text_columns = [configured_text_column] if configured_text_column else []
        feature_weights = None

        if allow_interactive_fallback and sys.stdin.isatty():
            entered_dataset = input(f"Dataset name for metadata storage [{dataset_name}]: ").strip()
            if entered_dataset:
                dataset_name = entered_dataset
                metadata_file = _metadata_path(dataset_name)

            text_columns = _prompt_free_text_columns(df, threshold)
            if text_columns:
                print(f"Confirmed free-text columns: {', '.join(text_columns)}")
            else:
                if configured_text_column:
                    print("No free-text columns confirmed interactively; falling back to configured text_column.")
                    text_columns = [configured_text_column]
                else:
                    print("No free-text columns confirmed interactively; continuing with tabular-only generation.")

            feature_weights = _prompt_feature_weights(df, set(text_columns))

            metadata_payload: dict[str, Any] = {
                "dataset_name": dataset_name,
                "free_text_columns": text_columns,
            }
            if feature_weights is not None:
                metadata_payload["feature_weights"] = feature_weights
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            metadata_file.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
            print(f"Saved interactive metadata to: {metadata_file}")
        else:
            print(f"Metadata JSON not found at {metadata_file}; using configured defaults.")

    text_columns = [col for col in text_columns if col]
    if not text_columns and configured_text_column:
        text_columns = [configured_text_column]

    unique_text_columns: list[str] = []
    for col in text_columns:
        if col not in unique_text_columns:
            unique_text_columns.append(col)

    missing_columns = [col for col in unique_text_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Selected text columns do not exist in dataset: {missing_columns}")

    if cli_feature_weights_path is not None:
        feature_weights = _load_feature_weights(cli_feature_weights_path)
        print(f"Using CLI feature weights from: {cli_feature_weights_path}")

    return unique_text_columns, feature_weights


def main() -> None:
    load_dotenv()
    args = parse_args()

    print("\n=== Synthetic Data Generation Pipeline ===")
    print("Loading configuration files...")
    print(f"- tabular config: {args.tabular_config}")
    print(f"- llm config: {args.llm_config}")
    print(f"- prompts config: {args.prompts_config}")

    tabular_cfg = _read_yaml(args.tabular_config)
    llm_cfg = _merge_with_cli(_read_yaml(args.llm_config), args)
    prompt_cfg = _read_yaml(args.prompts_config)

    input_csv = args.input_csv or Path(tabular_cfg["input_csv"])
    output_csv = args.output_csv or Path(tabular_cfg["output_csv"])

    print("\nResolved runtime settings:")
    print(f"- input_csv: {input_csv}")
    print(f"- output_csv: {output_csv}")
    print(f"- feature_weights_json: {args.feature_weights_json or '(none)'}")
    print(f"- text_column: {tabular_cfg.get('text_column', '(none)')}")
    neighbor_k = int(tabular_cfg['knn_k'])
    print(f"- shared neighbor_k (from tabular.knn_k): {neighbor_k}")
    llm_cfg['k_neighbors'] = neighbor_k
    print(f"- tabular epsilon: {tabular_cfg['epsilon']}")
    print(f"- enforce_min_max_values: {tabular_cfg['enforce_min_max_values']}")
    print(f"- llm model: {llm_cfg['model']}")
    print(f"- llm provider: {llm_cfg.get('provider', 'auto')}")
    print(f"- llm gpu_memory_utilization: {llm_cfg['gpu_memory_utilization']}")
    print(f"- llm max_model_len: {llm_cfg['max_model_len']}")
    print(f"- llm batch_size: {llm_cfg.get('batch_size', 1)}")
    print(f"- llm temperature/top_p: {llm_cfg['temperature']}/{llm_cfg['top_p']}")
    print(f"- llm max_tokens: {llm_cfg['max_tokens']}")
    max_words_generation = int(llm_cfg.get('max_words_generation', llm_cfg.get('max_words', 80)))
    max_words_reader = int(llm_cfg.get('max_words_reader', prompt_cfg.get('max_words_reader', 250)))
    print(f"- llm max_words_generation: {max_words_generation}")
    print(f"- llm max_words_reader: {max_words_reader}")
    print(f"- llm parallel_api_calls: {bool(llm_cfg.get('parallel_api_calls', False))}")
    print(f"- llm api_parallel_shards: {int(llm_cfg.get('api_parallel_shards', 5))}")
    distance_alpha = float(llm_cfg.get("distance_alpha", 1.0))
    distance_beta_default = float(llm_cfg.get("distance_beta_default", 0.2))
    distance_beta_by_column = llm_cfg.get("distance_beta_by_column", {})
    if not isinstance(distance_beta_by_column, dict):
        raise ValueError("llm.distance_beta_by_column must be a mapping of text_column -> beta weight.")
    print(f"- distance alpha (tabular block): {distance_alpha}")
    print(f"- distance beta default (text blocks): {distance_beta_default}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    print(f"\nLoading input dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded dataframe with shape: {df.shape}")

    configured_text_column = tabular_cfg.get("text_column")
    dataset_name = args.dataset_name or _dataset_name_from_path(input_csv)
    text_columns, feature_weights = _resolve_metadata(
        df,
        dataset_name=dataset_name,
        configured_text_column=configured_text_column,
        threshold=float(args.free_text_threshold),
        cli_feature_weights_path=args.feature_weights_json,
        configured_json_path=tabular_cfg.get("json_path"),
    )
    print(f"- active text_columns: {text_columns}")

    total_rows = len(df)
    if bool(llm_cfg.get("parallel_api_calls", False)):
        parallel_parameter_name = "api_parallel_shards"
        parallel_threads = max(1, int(llm_cfg.get(parallel_parameter_name, 1)))
        if parallel_threads > total_rows:
            raise ValueError(
                f"Please check the parallel {parallel_parameter_name}; it is currently larger than the dataset size."
            )

    print("\n[1/3] Running tabular data generation...")
    tabular_result = run_tabular_generation(
        df,
        tabular_cfg,
        text_columns=text_columns,
        feature_weights=feature_weights,
    )
    print(
        "Tabular generation complete. "
        f"Synthetic shape: {tabular_result.synthetic_data.shape}; "
        f"KNN index shape: {tabular_result.nn_idx.shape}"
    )

    generated = tabular_result.synthetic_data.copy()
    output_model_name = llm_cfg["model"]
    text_vector_blocks: dict[str, dict[str, pd.DataFrame]] = {}
    stage_nn_idx = tabular_result.nn_idx
    if text_columns:
        server_args = SimpleNamespace(
            model=llm_cfg["model"],
            provider=llm_cfg.get("provider", "auto"),
            utilization=float(llm_cfg["gpu_memory_utilization"]),
            max_len=int(llm_cfg["max_model_len"]),
            enforce_eager=bool(llm_cfg.get("enforce_eager", False) or args.enforce_eager),
        )
        llm_server = SingleGPUServer(server_args)
        output_model_name = llm_server.model_name
        try:
            print("\n[2/3] Running LLM free-text generation...")
            for stage_idx, text_column in enumerate(text_columns, start=1):
                print(f"\n  -> Free-text stage {stage_idx}/{len(text_columns)} for column '{text_column}'")
                if text_vector_blocks:
                    weighted_blocks: list[tuple[pd.DataFrame, pd.DataFrame, float]] = []
                    for prev_col, block in text_vector_blocks.items():
                        beta = float(distance_beta_by_column.get(prev_col, distance_beta_default))
                        weighted_blocks.append((block["real"], block["synthetic"], beta))
                    stage_nn_idx, _ = knn_retrieval_with_text_blocks(
                        tabular_result.real_z,
                        tabular_result.z_correlated,
                        k=int(tabular_cfg["knn_k"]),
                        alpha=distance_alpha,
                        text_blocks=weighted_blocks,
                        feature_weights=feature_weights,
                    )
                    print(
                        "     Updated KNN with blended distance: "
                        f"alpha={distance_alpha}, text_blocks={len(weighted_blocks)}"
                    )
                else:
                    stage_nn_idx = tabular_result.nn_idx
                    print("     Using tabular-only KNN for first text stage.")

                generated = generate_free_text_column(
                    llm_server=llm_server,
                    original_df=df,
                    synthetic_df=generated,
                    nn_idx=stage_nn_idx,
                    llm_cfg=llm_cfg,
                    prompt_cfg=prompt_cfg,
                    text_column=text_column,
                )

                hash_dim = int(llm_cfg.get("text_vector_hash_dim", 8))
                real_text_vectors = prepare_text_vector_features(df[text_column], hash_dim=hash_dim)
                synthetic_text_vectors = prepare_text_vector_features(generated[text_column], hash_dim=hash_dim)
                text_vector_blocks[text_column] = {
                    "real": real_text_vectors,
                    "synthetic": synthetic_text_vectors,
                }
                print(
                    "     Prepared text vectors for distance integration: "
                    f"column={text_column}, real={real_text_vectors.shape}, synthetic={synthetic_text_vectors.shape}"
                )
        finally:
            if llm_server.provider == "local":
                llm_server.cleanup()
    else:
        print("\n[2/3] Skipping LLM free-text generation (no text columns configured).")

    print("[3/3] Saving generated dataset...")
    output_csv = _with_model_suffix(output_csv, output_model_name)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output_csv, index=False)
    print(f"✅ Saved synthetic dataset to: {output_csv}")

    run_details = {
        "dataset_name": dataset_name,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "text_columns": text_columns,
        "feature_weights": feature_weights,
        "tabular_config": tabular_cfg,
        "llm_config": llm_cfg,
        "prompts_config": prompt_cfg,
        "resolved_hyperparameters": {
            "knn_k": int(tabular_cfg["knn_k"]),
            "epsilon": tabular_cfg["epsilon"],
            "enforce_min_max_values": bool(tabular_cfg["enforce_min_max_values"]),
            "distance_alpha": distance_alpha,
            "distance_beta_default": distance_beta_default,
            "distance_beta_by_column": distance_beta_by_column,
            "max_words_generation": max_words_generation,
            "max_words_reader": max_words_reader,
            "parallel_api_calls": bool(llm_cfg.get("parallel_api_calls", False)),
            "api_parallel_shards": int(llm_cfg.get("api_parallel_shards", 5)),
        },
    }
    log_path = _write_run_log(output_csv, run_details)
    print(f"📝 Saved run hyperparameter log to: {log_path}")


if __name__ == "__main__":
    main()
