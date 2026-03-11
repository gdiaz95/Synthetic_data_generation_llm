from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import re
import pandas as pd
import yaml
from dotenv import load_dotenv
import sys

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from source.llm.server_model import SingleGPUServer
from source.pipelines.run_free_text import generate_free_text_column
from source.pipelines.run_tabular import run_tabular_generation


DEFAULT_CONFIG_DIR = Path("configs")


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(data)}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic tabular dataset with an LLM-generated free-text column.",
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
    print(f"- text_column: {tabular_cfg.get('text_column', 'job_description')}")
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
    print(f"- llm max_words: {llm_cfg['max_words']}")
    print(f"- llm parallel_api_calls: {bool(llm_cfg.get('parallel_api_calls', False))}")
    print(f"- llm api_parallel_shards: {int(llm_cfg.get('api_parallel_shards', 5))}")
    print(f"- prompt neighbor_excerpt_words: {prompt_cfg['neighbor_excerpt_words']}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    print(f"\nLoading input dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded dataframe with shape: {df.shape}")

    text_column = tabular_cfg.get("text_column", "job_description")
    if text_column not in df.columns:
        raise ValueError(f"Expected text column '{text_column}' in input CSV.")

    total_rows = len(df)
    if bool(llm_cfg.get("parallel_api_calls", False)):
        parallel_parameter_name = "api_parallel_shards"
        parallel_threads = max(1, int(llm_cfg.get(parallel_parameter_name, 1)))
        if parallel_threads > total_rows:
            raise ValueError(
                f"Please check the parallel {parallel_parameter_name}; it is currently larger than the dataset size."
            )

    feature_weights = _load_feature_weights(args.feature_weights_json)

    print("\n[1/3] Running tabular data generation...")
    tabular_result = run_tabular_generation(
        df,
        tabular_cfg,
        text_column=text_column,
        feature_weights=feature_weights,
    )
    print(
        "Tabular generation complete. "
        f"Synthetic shape: {tabular_result.synthetic_data.shape}; "
        f"KNN index shape: {tabular_result.nn_idx.shape}"
    )

    server_args = SimpleNamespace(
        model=llm_cfg["model"],
        provider=llm_cfg.get("provider", "auto"),
        utilization=float(llm_cfg["gpu_memory_utilization"]),
        max_len=int(llm_cfg["max_model_len"]),
        enforce_eager=bool(llm_cfg.get("enforce_eager", False) or args.enforce_eager),
    )

    llm_server = SingleGPUServer(server_args)
    try:
        print("\n[2/3] Running LLM free-text generation...")
        generated = generate_free_text_column(
            llm_server=llm_server,
            original_df=df,
            synthetic_df=tabular_result.synthetic_data,
            nn_idx=tabular_result.nn_idx,
            llm_cfg=llm_cfg,
            prompt_cfg=prompt_cfg,
            text_column=text_column,
        )
    finally:
        if llm_server.provider == "local":
            llm_server.cleanup()

    print("[3/3] Saving generated dataset...")
    output_csv = _with_model_suffix(output_csv, llm_server.model_name)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output_csv, index=False)
    print(f"✅ Saved synthetic dataset to: {output_csv}")


if __name__ == "__main__":
    main()
