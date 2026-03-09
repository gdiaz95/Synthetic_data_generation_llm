from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from tqdm import tqdm

from source.llm.server_model import SingleGPUServer



META_PHRASES = [
    "[end]",
    "to ensure adherence",
    "word count",
    "fits constraints",
    "ignore previous",
    "return only",
    "here's your requested",
]


def sanitize_neighbor_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    safe_lines: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(phrase in lower for phrase in META_PHRASES):
            continue
        safe_lines.append(line)

    sanitized = " ".join(safe_lines).strip()
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized


def clean_generated_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    lower = cleaned.lower()
    for phrase in META_PHRASES:
        idx = lower.find(phrase)
        if idx != -1:
            cleaned = cleaned[:idx].strip()
            lower = cleaned.lower()

    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def looks_like_meta_output(text: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    return any(phrase in lower for phrase in META_PHRASES)

def cap_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", (text or "").strip())
    return " ".join(words[:max_words]).strip()


def build_prompt_from_structured_with_neighbors(
    fields: dict[str, Any],
    prompt_template: str,
    max_words: int,
    text_column: str,
    neighbor_rows: list[dict[str, Any]] | None = None,
    max_neighbors: int = 3,
    neighbor_excerpt_words: int = 20,
) -> str:
    clean = {
        key: value
        for key, value in fields.items()
        if key != text_column and value is not None and pd.notna(value)
    }

    snippets: list[str] = []
    if neighbor_rows:
        for idx, row in enumerate(neighbor_rows[:max_neighbors], start=1):
            row_fields = {
                key: value
                for key, value in row.items()
                if key != text_column and value is not None and pd.notna(value)
            }
            excerpt = cap_words(str(row.get(text_column, "")), neighbor_excerpt_words)
            if excerpt:
                snippets.append(
                    f"- Neighbor {idx}\\n"
                    f"  profile: {row_fields}\\n"
                    f"  {text_column}: {excerpt}"
                )

    neighbor_block = "\n".join(snippets) if snippets else "(none)"

    return prompt_template.format(
        max_words=max_words,
        text_column=text_column,
        structured_fields=clean,
        neighbor_block=neighbor_block,
    ).strip()


def generate_free_text_column(
    llm_server: SingleGPUServer,
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    nn_idx,
    llm_cfg: dict,
    prompt_cfg: dict,
    text_column: str,
) -> pd.DataFrame:
    print(f"- Starting free-text generation stage for column '{text_column}'...")
    generated = synthetic_df.copy()
    generated[text_column] = ""

    total_rows = len(generated)
    batch_size = max(1, int(llm_cfg.get("batch_size", 1)))
    generation_config = {
        "temperature": float(llm_cfg["temperature"]),
        "top_p": float(llm_cfg["top_p"]),
        "max_tokens": int(llm_cfg["max_tokens"]),
        "stop_sequences": llm_cfg.get("stop_sequences"),
    }
    print(f"- Total rows to generate: {total_rows}")
    print(f"- Batch size: {batch_size}")
    print(f"- K neighbors used in prompts: {int(llm_cfg['k_neighbors'])}")
    print(
        "- Generation config: "
        f"temperature={generation_config['temperature']}, "
        f"top_p={generation_config['top_p']}, "
        f"max_tokens={generation_config['max_tokens']}, "
        f"max_words={int(llm_cfg['max_words'])}"
    )

    batch_log_every = max(0, int(llm_cfg.get("batch_log_every", 0)))
    if batch_log_every:
        print(f"- Batch detail logging enabled every {batch_log_every} batch(es).")
    else:
        print("- Batch detail logging disabled (set llm.batch_log_every > 0 to enable).")

    parallel_api_calls = bool(llm_cfg.get("parallel_api_calls", False))
    api_parallel_shards = max(1, int(llm_cfg.get("api_parallel_shards", 5)))
    if llm_server.provider in {"gemini", "openai"} and parallel_api_calls:
        print(
            "- Parallel API mode: enabled "
            f"(provider={llm_server.provider}, shards={api_parallel_shards})."
        )
    else:
        print(
            "- Parallel API mode: disabled "
            "(either flag is false or provider is local)."
        )

    def _build_prompts_for_indices(batch_indices: list[int]) -> list[str]:
        prompts: list[str] = []
        for i in batch_indices:
            neighbor_pos = nn_idx[i, : int(llm_cfg["k_neighbors"])]
            neighbor_rows = original_df.iloc[neighbor_pos].to_dict("records")
            sanitized_neighbor_rows: list[dict[str, Any]] = []
            for row in neighbor_rows:
                sanitized = sanitize_neighbor_text(str(row.get(text_column, "")))
                if not sanitized:
                    continue
                row_copy = dict(row)
                row_copy[text_column] = sanitized
                sanitized_neighbor_rows.append(row_copy)

            prompt = build_prompt_from_structured_with_neighbors(
                fields=generated.iloc[i].to_dict(),
                prompt_template=prompt_cfg["free_text_prompt"],
                max_words=int(llm_cfg["max_words"]),
                text_column=text_column,
                neighbor_rows=sanitized_neighbor_rows,
                max_neighbors=int(llm_cfg["k_neighbors"]),
                neighbor_excerpt_words=int(prompt_cfg["neighbor_excerpt_words"]),
            )
            prompts.append(prompt)
        return prompts

    def _process_batch(batch_indices: list[int]) -> dict[int, str]:
        prompts = _build_prompts_for_indices(batch_indices)
        responses = llm_server.generate_batch(prompts, config=generation_config)
        row_updates: dict[int, str] = {}
        for i, response_text in zip(batch_indices, responses):
            cleaned_response = clean_generated_text(response_text)
            if looks_like_meta_output(cleaned_response):
                cleaned_response = ""
            row_updates[i] = cap_words(
                cleaned_response,
                int(llm_cfg["max_words"]),
            )
        return row_updates

    def _run_sequential() -> pd.DataFrame:
        progress = tqdm(
            range(0, total_rows, batch_size),
            desc=f"Generating {text_column}",
            dynamic_ncols=True,
        )

        for batch_num, batch_start in enumerate(progress, start=1):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_indices = list(range(batch_start, batch_end))
            row_updates = _process_batch(batch_indices)

            if batch_log_every and batch_num % batch_log_every == 0:
                progress.write(
                    "  • Batch "
                    f"{batch_num}: rows [{batch_start}:{batch_end}] "
                    f"-> received {len(batch_indices)} response(s)."
                )

            for i, text in row_updates.items():
                generated.at[generated.index[i], text_column] = text

            time.sleep(float(llm_cfg["sleep_sec"]))

        return generated

    def _run_parallel_api() -> pd.DataFrame:
        shard_indices = [
            list(range(start, total_rows, api_parallel_shards))
            for start in range(api_parallel_shards)
        ]
        shard_indices = [indices for indices in shard_indices if indices]

        progress_bars = []
        for shard_id, indices in enumerate(shard_indices):
            bar = tqdm(
                total=len(indices),
                desc=f"Shard {shard_id + 1}/{len(shard_indices)}",
                position=shard_id,
                leave=True,
                dynamic_ncols=True,
            )
            progress_bars.append(bar)

        def _run_shard(shard_id: int, indices: list[int]) -> tuple[int, dict[int, str]]:
            shard_updates: dict[int, str] = {}
            for batch_num, offset in enumerate(range(0, len(indices), batch_size), start=1):
                batch_indices = indices[offset : offset + batch_size]
                row_updates = _process_batch(batch_indices)
                shard_updates.update(row_updates)
                progress_bars[shard_id].update(len(batch_indices))
                if batch_log_every and batch_num % batch_log_every == 0:
                    progress_bars[shard_id].write(
                        f"  • Shard {shard_id + 1} batch {batch_num}: "
                        f"generated {len(batch_indices)} row(s)."
                    )
                time.sleep(float(llm_cfg["sleep_sec"]))
            return shard_id, shard_updates

        with ThreadPoolExecutor(max_workers=len(shard_indices)) as executor:
            futures = [
                executor.submit(_run_shard, shard_id, indices)
                for shard_id, indices in enumerate(shard_indices)
            ]
            for future in as_completed(futures):
                _, shard_updates = future.result()
                for i, text in shard_updates.items():
                    generated.at[generated.index[i], text_column] = text

        for bar in progress_bars:
            bar.close()

        return generated
    if llm_server.provider in {"gemini", "openai"} and parallel_api_calls:
        generated = _run_parallel_api()
    else:
        generated = _run_sequential()

    print("- Free-text generation complete.")

    return generated
