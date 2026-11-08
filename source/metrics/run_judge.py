"""Run the LLM-as-a-judge step (column-level and/or whole-row) against
already-generated real/synthetic data, for a single (dataset, generator,
judge) triple.

The paper evaluates every (dataset, generator) pair with four separate judge
models. `source/metrics/run_structured_metrics.py` does not run any judge
itself (QA/Frechet/MMD/joint-discriminator/TSTR only) -- this script replicates
its exact data loading, row capping, and train/holdout split (same max_rows,
same random_state) and calls evaluate_llm_judge() directly, once per judge,
so all four judges' results can be computed independently of each other and
of the non-judge metrics.

Output: reports/judges/{dataset}_{generator}_{judge}_judge.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from source.metrics.llm_judge import evaluate_llm_judge


DEFAULT_CONFIG = Path("configs/metrics.yml")
DEFAULT_LLM_CONFIG = Path("configs/llm_generation.yml")
DEFAULT_METADATA_ROOT = Path(".metadata")
OUTPUT_DIR = Path("reports/judges")


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(data)}")
    return data


def _safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-") or "model"


def _resolve_path(template: str, *, dataset_name: str, model: str) -> Path:
    safe = _safe_model_name(model)
    return Path(template.format(dataset_name=dataset_name, generator_model=safe, model=safe))


def _load_metadata(dataset_name: str) -> dict[str, Any] | None:
    path = DEFAULT_METADATA_ROOT / dataset_name / "metadata.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata file must be a JSON object: {path}")
    return payload


def _confirm_llm_billing(auto_yes: bool) -> bool:
    prompt = "LLM judge evaluation can trigger external API billing. Continue? [y/N]: "
    if auto_yes:
        print(f"[--yes] Auto-approving: {prompt.strip()}")
        return True
    if not sys.stdin.isatty():
        print("Non-interactive session detected; defaulting to NO.")
        return False
    return input(prompt).strip().lower() in {"y", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the LLM-as-a-judge step against already-generated data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument(
        "--generator-model",
        type=str,
        required=True,
        help="Identifies which synthetic file to judge, e.g. 'gemini-3.1-flash-lite', "
        "'gemini-3.1-flash-lite_k0', or 'Qwen/Qwen2.5-7B-Instruct_great' for the GReaT baseline.",
    )
    parser.add_argument("--judge-model", type=str, required=True)
    parser.add_argument("--judge-provider", type=str, default="auto", choices=["auto", "local", "gemini", "openai"])
    parser.add_argument(
        "--whole-row-only",
        action="store_true",
        help="Skip per-column judging (pass free_text_columns=[] to evaluate_llm_judge); "
        "use for the k=0 ablation and the GReaT-baseline comparison, which only need whole-row accuracy.",
    )
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = _read_yaml(args.config)

    dataset_name = args.dataset_name
    model = args.generator_model
    holdout_fraction = float(cfg.get("holdout_fraction", 0.2))
    random_seed = int(cfg.get("random_state", 42))

    real_template = cfg["real_data_path"]
    synthetic_template = cfg["synthetic_data_path"]
    real_path = _resolve_path(real_template, dataset_name=dataset_name, model=model)
    synthetic_path = _resolve_path(synthetic_template, dataset_name=dataset_name, model=model)

    if not real_path.exists():
        raise FileNotFoundError(f"Real data not found at: {real_path}")
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Synthetic data not found at: {synthetic_path}")

    metadata = _load_metadata(dataset_name)
    text_columns = [str(c) for c in (metadata.get("free_text_columns") or [])] if metadata else []
    if args.whole_row_only:
        print(f"--whole-row-only: skipping per-column judging (dataset's free-text columns: {text_columns})")
        judged_text_columns: list[str] = []
    else:
        judged_text_columns = text_columns

    print(f"Dataset        : {dataset_name}")
    print(f"Generator      : {model}")
    print(f"Judge          : {args.judge_model} ({args.judge_provider})")
    print(f"Real           : {real_path}")
    print(f"Synthetic      : {synthetic_path}")
    print(f"Text columns   : {judged_text_columns}")

    fallback_judge_cfg = _read_yaml(DEFAULT_LLM_CONFIG) if DEFAULT_LLM_CONFIG.exists() else {}
    base_judge_cfg = cfg.get("llm_judge", {}) or {}
    judge_cfg = {
        **fallback_judge_cfg,
        **base_judge_cfg,
        "model": args.judge_model,
        "provider": args.judge_provider,
    }
    if args.gpu_index is not None:
        judge_cfg["gpu_index"] = args.gpu_index

    if not _confirm_llm_billing(args.yes):
        print("LLM billing permission denied; aborting.")
        return

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

    real_train, real_holdout = train_test_split(
        real_df, test_size=holdout_fraction, random_state=random_seed
    )
    del real_train  # only the holdout split is used for judging, matching run_structured_metrics.py

    print("Evaluating LLM-as-a-judge metrics...")
    judge_data = evaluate_llm_judge(
        real_holdout_df=real_holdout.reset_index(drop=True),
        synthetic_df=synthetic_df.reset_index(drop=True),
        free_text_columns=judged_text_columns,
        judge_cfg=judge_cfg,
    )

    safe_gen = _safe_model_name(model)
    safe_judge = _safe_model_name(args.judge_model)
    output_path = args.output or (OUTPUT_DIR / f"{dataset_name}_{safe_gen}_{safe_judge}_judge.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "generator_model": model,
                "whole_row_only": args.whole_row_only,
                "llm_judge": judge_data,
            },
            indent=2,
        )
    )
    print(f"\nSaved: {output_path}")
    if "whole_row" in judge_data:
        print(f"Whole-row accuracy: {judge_data['whole_row'].get('accuracy')}")


if __name__ == "__main__":
    main()
