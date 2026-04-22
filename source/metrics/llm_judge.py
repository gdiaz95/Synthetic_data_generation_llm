from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from source.llm.server_model import SingleGPUServer


REAL_LABEL = "real"
SYNTHETIC_LABEL = "synthetic"


@dataclass
class LLMJudgeSummary:
    task: str
    target: str
    model: str
    provider: str
    accuracy: float
    n_examples: int
    n_real: int
    n_synthetic: int
    correct: int
    incorrect: int
    predicted_real: int
    predicted_synthetic: int
    true_real_predicted_real: int
    true_real_predicted_synthetic: int
    true_synthetic_predicted_real: int
    true_synthetic_predicted_synthetic: int


def _default_generation_config(batch_size: int, thinking_budget: int | None = None) -> dict[str, Any]:
    # Thinking models count thinking tokens against max_output_tokens, so 48 is
    # exhausted before any visible output is produced. Use a larger budget.
    max_tokens = 4096 if thinking_budget is not None else 48
    config: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "batch_size": max(1, int(batch_size)),
        "stop_sequences": ["\n\n"],
    }
    if thinking_budget is not None:
        config["thinking_budget"] = int(thinking_budget)
    return config


def _build_server(judge_cfg: dict[str, Any]) -> SingleGPUServer:
    model = str(judge_cfg.get("model", "")).strip()
    if not model:
        raise ValueError("llm_judge.model must be configured when llm_judge.enabled=true.")

    provider = str(judge_cfg.get("provider", "auto")).strip() or "auto"
    args = SimpleNamespace(
        model=model,
        provider=provider,
        utilization=float(judge_cfg.get("gpu_memory_utilization", 0.8)),
        max_len=int(judge_cfg.get("max_model_len", 4096)),
        enforce_eager=bool(judge_cfg.get("enforce_eager", False)),
    )
    return SingleGPUServer(args)


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _row_to_payload(row: pd.Series) -> dict[str, Any]:
    return {str(key): _normalize_value(value) for key, value in row.items()}


def _extract_label(raw_text: str) -> str | None:
    if not raw_text:
        return None

    stripped = raw_text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            label = str(parsed.get("label", "")).strip().lower()
            if label in {REAL_LABEL, SYNTHETIC_LABEL}:
                return label
    except json.JSONDecodeError:
        pass

    match = re.search(r"\b(real|synthetic)\b", stripped.lower())
    if match:
        return match.group(1)
    return None


def _judge_prompt_for_text(column_name: str, text_value: str) -> str:
    return (
        "You are evaluating whether a single text entry is from the real dataset or the "
        "synthetic dataset.\n"
        f"Column: {column_name}\n"
        "Rules:\n"
        "- Reply with JSON only.\n"
        '- Use exactly one label: "real" or "synthetic".\n'
        '- Format: {"label":"real"} or {"label":"synthetic"}.\n'
        "- Do not add explanations.\n\n"
        "Text entry:\n"
        f"{text_value}"
    )


def _judge_prompt_for_row(row_payload: dict[str, Any]) -> str:
    row_json = json.dumps(row_payload, ensure_ascii=False, sort_keys=True)
    return (
        "You are evaluating whether one full table row is from the real dataset or the "
        "synthetic dataset.\n"
        "Rules:\n"
        "- Reply with JSON only.\n"
        '- Use exactly one label: "real" or "synthetic".\n'
        '- Format: {"label":"real"} or {"label":"synthetic"}.\n'
        "- Do not add explanations.\n\n"
        "Row:\n"
        f"{row_json}"
    )


def _balanced_sample(
    real_records: list[Any],
    synthetic_records: list[Any],
    *,
    max_examples_per_class: int,
    random_state: int,
) -> list[tuple[Any, str]]:
    rng = np.random.default_rng(random_state)
    real_count = min(len(real_records), max_examples_per_class)
    synthetic_count = min(len(synthetic_records), max_examples_per_class)

    if real_count == 0 or synthetic_count == 0:
        raise ValueError("LLM judge requires at least one real and one synthetic example.")

    real_idx = rng.choice(len(real_records), size=real_count, replace=False)
    synthetic_idx = rng.choice(len(synthetic_records), size=synthetic_count, replace=False)

    combined: list[tuple[Any, str]] = [
        (real_records[int(idx)], REAL_LABEL) for idx in real_idx
    ] + [
        (synthetic_records[int(idx)], SYNTHETIC_LABEL) for idx in synthetic_idx
    ]

    order = rng.permutation(len(combined))
    return [combined[int(idx)] for idx in order]


def _run_prompts(
    llm_server: SingleGPUServer,
    prompts: list[str],
    *,
    batch_size: int,
    thinking_budget: int | None = None,
    progress_label: str | None = None,
) -> list[str]:
    if not prompts:
        return []

    generation_config = _default_generation_config(batch_size=batch_size, thinking_budget=thinking_budget)
    responses: list[str] = []
    total_batches = math.ceil(len(prompts) / batch_size)
    for batch_index, start in enumerate(range(0, len(prompts), batch_size), start=1):
        batch_prompts = prompts[start : start + batch_size]
        responses.extend(llm_server.generate_batch(batch_prompts, config=generation_config))
        if progress_label:
            _print_progress_bar(progress_label, batch_index, total_batches)
    return responses


def _print_progress_bar(label: str, current: int, total: int) -> None:
    width = 24
    filled = width if total <= 0 else int(width * current / total)
    bar = "=" * filled + "-" * max(0, width - filled)
    end = "\n" if current >= total else "\r"
    sys.stdout.write(f"[llm-judge] {label}: |{bar}| {current}/{total}{end}")
    sys.stdout.flush()


def _summarize_predictions(
    *,
    task: str,
    target: str,
    model: str,
    provider: str,
    truth_labels: list[str],
    predicted_labels: list[str | None],
) -> dict[str, Any]:
    n_examples = len(truth_labels)
    n_real = sum(label == REAL_LABEL for label in truth_labels)
    n_synthetic = sum(label == SYNTHETIC_LABEL for label in truth_labels)
    predicted_real = sum(label == REAL_LABEL for label in predicted_labels)
    predicted_synthetic = sum(label == SYNTHETIC_LABEL for label in predicted_labels)
    true_real_predicted_real = sum(
        truth == REAL_LABEL and pred == REAL_LABEL
        for truth, pred in zip(truth_labels, predicted_labels, strict=True)
    )
    true_real_predicted_synthetic = sum(
        truth == REAL_LABEL and pred == SYNTHETIC_LABEL
        for truth, pred in zip(truth_labels, predicted_labels, strict=True)
    )
    true_synthetic_predicted_real = sum(
        truth == SYNTHETIC_LABEL and pred == REAL_LABEL
        for truth, pred in zip(truth_labels, predicted_labels, strict=True)
    )
    true_synthetic_predicted_synthetic = sum(
        truth == SYNTHETIC_LABEL and pred == SYNTHETIC_LABEL
        for truth, pred in zip(truth_labels, predicted_labels, strict=True)
    )
    correct = sum(
        truth == pred for truth, pred in zip(truth_labels, predicted_labels, strict=True)
    )
    incorrect = n_examples - correct
    accuracy = float(correct / n_examples) if n_examples else float("nan")

    summary = asdict(
        LLMJudgeSummary(
            task=task,
            target=target,
            model=model,
            provider=provider,
            accuracy=accuracy,
            n_examples=n_examples,
            n_real=n_real,
            n_synthetic=n_synthetic,
            correct=correct,
            incorrect=incorrect,
            predicted_real=predicted_real,
            predicted_synthetic=predicted_synthetic,
            true_real_predicted_real=true_real_predicted_real,
            true_real_predicted_synthetic=true_real_predicted_synthetic,
            true_synthetic_predicted_real=true_synthetic_predicted_real,
            true_synthetic_predicted_synthetic=true_synthetic_predicted_synthetic,
        )
    )
    summary["failed_to_parse"] = int(sum(pred is None for pred in predicted_labels))
    return summary


def evaluate_llm_judge(
    *,
    real_holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    free_text_columns: list[str],
    judge_cfg: dict[str, Any],
) -> dict[str, Any]:
    batch_size = max(1, int(judge_cfg.get("batch_size", 8)))
    max_examples_per_class = max(1, int(judge_cfg.get("max_examples_per_class", 128)))
    random_state = int(judge_cfg.get("random_state", 42))
    thinking_budget_raw = judge_cfg.get("thinking_budget")
    thinking_budget: int | None = int(thinking_budget_raw) if thinking_budget_raw is not None else None

    server = _build_server(judge_cfg)
    try:
        free_text_results: dict[str, Any] = {}

        for offset, column_name in enumerate(free_text_columns):
            if column_name not in real_holdout_df.columns or column_name not in synthetic_df.columns:
                free_text_results[column_name] = {
                    "error": f"Column '{column_name}' is missing in real holdout or synthetic data."
                }
                continue

            real_text = (
                real_holdout_df[column_name]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            synthetic_text = (
                synthetic_df[column_name]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            real_records = [value for value in real_text.tolist() if value]
            synthetic_records = [value for value in synthetic_text.tolist() if value]
            try:
                sampled = _balanced_sample(
                    real_records,
                    synthetic_records,
                    max_examples_per_class=max_examples_per_class,
                    random_state=random_state + offset,
                )
                prompts = [_judge_prompt_for_text(column_name, value) for value, _ in sampled]
                truth_labels = [label for _, label in sampled]
                raw_predictions = _run_prompts(
                    server,
                    prompts,
                    batch_size=batch_size,
                    thinking_budget=thinking_budget,
                    progress_label=f"{column_name}",
                )
                predicted_labels = [_extract_label(prediction) for prediction in raw_predictions]
                free_text_results[column_name] = _summarize_predictions(
                    task="free_text_column",
                    target=column_name,
                    model=server.model_name,
                    provider=server.provider,
                    truth_labels=truth_labels,
                    predicted_labels=predicted_labels,
                )
            except Exception as exc:
                free_text_results[column_name] = {"error": str(exc)}

        real_rows = [_row_to_payload(row) for _, row in real_holdout_df.iterrows()]
        synthetic_rows = [_row_to_payload(row) for _, row in synthetic_df.iterrows()]
        sampled_rows = _balanced_sample(
            real_rows,
            synthetic_rows,
            max_examples_per_class=max_examples_per_class,
            random_state=random_state + 10_000,
        )
        row_prompts = [_judge_prompt_for_row(payload) for payload, _ in sampled_rows]
        row_truth_labels = [label for _, label in sampled_rows]
        row_raw_predictions = _run_prompts(
            server,
            row_prompts,
            batch_size=batch_size,
            thinking_budget=thinking_budget,
            progress_label="whole_row",
        )
        row_predicted_labels = [_extract_label(prediction) for prediction in row_raw_predictions]
        whole_row_result = _summarize_predictions(
            task="whole_row",
            target="__whole_row__",
            model=server.model_name,
            provider=server.provider,
            truth_labels=row_truth_labels,
            predicted_labels=row_predicted_labels,
        )

        return {
            "judge_model": server.model_name,
            "judge_provider": server.provider,
            "usage": server.get_usage_summary(),
            "max_examples_per_class": max_examples_per_class,
            "free_text_columns": free_text_results,
            "whole_row": whole_row_result,
        }
    finally:
        if server.provider == "local":
            server.cleanup()
