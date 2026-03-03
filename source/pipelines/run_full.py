from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
from dotenv import load_dotenv

from source.llm.server_model import SingleGPUServer
from source.pipelines.run_free_text import generate_job_descriptions
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
        description="Generate synthetic tabular Adult dataset with LLM job descriptions.",
    )
    parser.add_argument("--tabular-config", type=Path, default=DEFAULT_CONFIG_DIR / "tabular.yml")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_CONFIG_DIR / "llm_generation.yml")
    parser.add_argument("--prompts-config", type=Path, default=DEFAULT_CONFIG_DIR / "prompts.yml")

    parser.add_argument("--input-csv", type=Path, help="Override input CSV path")
    parser.add_argument("--output-csv", type=Path, help="Override output CSV path")

    parser.add_argument("--model", type=str, help="Override generation model")
    parser.add_argument("--provider", choices=["auto", "local", "gemini", "openai"], help="Override backend provider")
    parser.add_argument("--gpu-memory-utilization", type=float, help="Override GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, help="Override max model length")

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


def main() -> None:
    load_dotenv()
    args = parse_args()

    tabular_cfg = _read_yaml(args.tabular_config)
    llm_cfg = _merge_with_cli(_read_yaml(args.llm_config), args)
    prompt_cfg = _read_yaml(args.prompts_config)

    input_csv = args.input_csv or Path(tabular_cfg["input_csv"])
    output_csv = args.output_csv or Path(tabular_cfg["output_csv"])

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "job_description" not in df.columns:
        raise ValueError("Expected column 'job_description' in input CSV.")

    tabular_result = run_tabular_generation(df, tabular_cfg)

    server_args = SimpleNamespace(
        model=llm_cfg["model"],
        provider=llm_cfg.get("provider", "auto"),
        utilization=float(llm_cfg["gpu_memory_utilization"]),
        max_len=int(llm_cfg["max_model_len"]),
    )

    llm_server = SingleGPUServer(server_args)
    try:
        generated = generate_job_descriptions(
            llm_server=llm_server,
            original_df=df,
            synthetic_df=tabular_result.synthetic_data,
            nn_idx=tabular_result.nn_idx,
            llm_cfg=llm_cfg,
            prompt_cfg=prompt_cfg,
        )
    finally:
        if llm_server.provider == "local":
            llm_server.cleanup()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output_csv, index=False)
    print(f"Saved synthetic dataset to: {output_csv}")


if __name__ == "__main__":
    main()
