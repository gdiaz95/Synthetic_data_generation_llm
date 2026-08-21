from __future__ import annotations

import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from tqdm import tqdm

from source.llm.server_model import SingleGPUServer, APIModelConnectionError



# Phrases that only ever show up as *trailing* meta-commentary the model
# appends after the real content — truncate from the phrase onward, keeping
# what came before it.
TRAILING_META_PHRASES = [
    "[end]",
    "to ensure adherence",
    "word count",
    "fits constraints",
    "ignore previous",
    # Leaked LLM instruction phrases found in neighbor data
    "focus on mimicking",
    "to sound human",
    "aim for consistency",
    "to apply, just",
]

# Phrases that show up as a *leading* preamble before the real content (e.g.
# "Here's your requested amenities list: <actual list>", or the model
# echoing the prompt's own "Return ONLY the {text_column} text." instruction
# before complying). Truncating to "everything before the phrase" would
# discard the entire real answer for these, so instead strip the preamble
# itself and keep what follows. This was the root cause of a near-total data
# loss on longer/more complex generations (e.g. ~99.75% of an "amenities"
# column silently blanked to empty strings) where a smaller/faster model
# tended to echo "return only" back before actually answering.
LEADING_META_PHRASES = [
    "return only",
    "here's your requested",
    "here goes:",
]

META_PHRASES = TRAILING_META_PHRASES + LEADING_META_PHRASES


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
    for phrase in TRAILING_META_PHRASES:
        idx = lower.find(phrase)
        if idx != -1:
            cleaned = cleaned[:idx].strip()
            lower = cleaned.lower()

    for phrase in LEADING_META_PHRASES:
        idx = lower.find(phrase)
        if idx != -1:
            after = cleaned[idx + len(phrase):]
            # A leading preamble is often followed by a few more words and a
            # colon before the real content (e.g. "...requested amenities
            # list: <content>") — skip up through the next colon if one
            # appears shortly after the phrase, not just immediately after it.
            colon_pos = after.find(":")
            if 0 <= colon_pos <= 40:
                after = after[colon_pos + 1:]
            cleaned = after.strip()
            lower = cleaned.lower()

    # Some models answer a list-shaped request (e.g. amenities) with a JSON
    # array instead of prose. Convert it to a plain comma-separated list
    # before the bracket-stripping below, which would otherwise treat the
    # whole response as one big "[annotation]" and wipe it entirely.
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                cleaned = ", ".join(parsed)
        except json.JSONDecodeError:
            pass

    # Strip small bracketed annotations (e.g. "[END]") that appear within an
    # otherwise-prose response — but never let this wipe out the entire
    # response (guards against any other case where the real content is
    # itself fully bracketed).
    stripped = re.sub(r"\[[^\]]*\]", "", cleaned).strip()
    cleaned = stripped if stripped else cleaned
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
    max_words_generation: int,
    text_column: str,
    neighbor_rows: list[dict[str, Any]] | None = None,
    max_neighbors: int = 3,
    max_words_reader: int = 250,
    target_min_words: int | None = None,
    target_max_words: int | None = None,
) -> str:
    clean: dict[str, Any] = {}
    for key, value in fields.items():
        if key == text_column or value is None:
            continue
        if not isinstance(value, (list, tuple, dict, set)) and pd.isna(value):
            continue
        if isinstance(value, str):
            clean[key] = cap_words(value, max_words_reader)
        else:
            clean[key] = value

    snippets: list[str] = []
    if neighbor_rows:
        for idx, row in enumerate(neighbor_rows[:max_neighbors], start=1):
            row_fields = {
                key: value
                for key, value in row.items()
                if key != text_column and value is not None and pd.notna(value)
            }
            excerpt = cap_words(str(row.get(text_column, "")), max_words_reader)
            if excerpt:
                snippets.append(
                    f"- Neighbor {idx}\n"
                    f"  profile: {row_fields}\n"
                    f"  {text_column}: {excerpt}"
                )

    neighbor_block = "\n".join(snippets) if snippets else "(none)"

    calibrated_target_min_words = max(
        1,
        int(target_min_words if target_min_words is not None else max_words_generation),
    )
    calibrated_target_max_words = max(
        calibrated_target_min_words,
        int(target_max_words if target_max_words is not None else max_words_generation),
    )

    return prompt_template.format(
        max_words_generation=max_words_generation,
        max_words_reader=max_words_reader,
        target_min_words=calibrated_target_min_words,
        target_max_words=calibrated_target_max_words,
        text_column=text_column,
        target_profile=clean,
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
    is_local = llm_server.provider == "local"
    batch_size = max(1, int(llm_cfg.get("batch_size", 1)))
    sleep_sec = float(llm_cfg.get("sleep_sec", 1.0))
    max_words_generation = int(
        llm_cfg.get("max_words_generation", llm_cfg.get("max_words", 80))
    )
    max_words_reader = int(llm_cfg.get("max_words_reader", prompt_cfg.get("max_words_reader", 250)))

    # Apply local-vLLM overrides when running on a local model.
    if is_local:
        if "batch_size_local" in llm_cfg:
            batch_size = max(1, int(llm_cfg["batch_size_local"]))
        if "sleep_sec_local" in llm_cfg:
            sleep_sec = float(llm_cfg["sleep_sec_local"])

    # Adjust parameters for models that don't support certain values
    temp_value = float(llm_cfg["temperature"])
    model_name = llm_cfg.get("model", "").lower()
    is_newer_model = any(newer_model in model_name for newer_model in ["gpt-5", "gpt-o1"])

    if is_newer_model:
        temp_value = 1.0

    max_tokens = int(llm_cfg["max_tokens"])
    if is_local and "max_tokens_local" in llm_cfg:
        max_tokens = int(llm_cfg["max_tokens_local"])

    generation_config = {
        "temperature": temp_value,
        "max_tokens": max_tokens,
        "stop_sequences": llm_cfg.get("stop_sequences"),
    }

    # Newer models don't support top_p
    if not is_newer_model:
        generation_config["top_p"] = float(llm_cfg["top_p"])

    print(f"- Total rows to generate: {total_rows}")
    print(f"- Batch size: {batch_size}" + (" [local override]" if is_local and "batch_size_local" in llm_cfg else ""))
    print(f"- Sleep between batches: {sleep_sec}s" + (" [local override]" if is_local and "sleep_sec_local" in llm_cfg else ""))
    print(f"- K neighbors used in prompts: {int(llm_cfg['k_neighbors'])}")
    config_str = (
        f"temperature={generation_config['temperature']}, "
        f"max_tokens={generation_config['max_tokens']}" + (" [local override]" if is_local and "max_tokens_local" in llm_cfg else "")
    )
    if "top_p" in generation_config:
        config_str += f", top_p={generation_config['top_p']}"
    print(
        "- Generation config: "
        f"{config_str}, "
        f"max_words_generation={max_words_generation}, "
        f"max_words_reader={max_words_reader}"
    )

    batch_log_every = max(0, int(llm_cfg.get("batch_log_every", 0)))
    if batch_log_every:
        print(f"- Batch detail logging enabled every {batch_log_every} batch(es).")
    else:
        print("- Batch detail logging disabled (set llm.batch_log_every > 0 to enable).")

    parallel_api_calls = bool(llm_cfg.get("parallel_api_calls", False))
    api_parallel_shards = max(1, int(llm_cfg.get("api_parallel_shards", 5)))
    rerun_failed_rows = bool(llm_cfg.get("rerun_failed_rows", True))
    failed_row_retry_passes = max(0, int(llm_cfg.get("failed_row_retry_passes", 1)))
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
    print(
        "- Failed-row rerun: "
        f"{'enabled' if rerun_failed_rows else 'disabled'} "
        f"(passes={failed_row_retry_passes})."
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

            neighbor_lengths = [
                len(re.findall(r"\S+", str(row.get(text_column, "")).strip()))
                for row in sanitized_neighbor_rows
                if str(row.get(text_column, "")).strip()
            ]
            if neighbor_lengths:
                avg_neighbor_words = sum(neighbor_lengths) / len(neighbor_lengths)
                std_neighbor_words = statistics.pstdev(neighbor_lengths)
            else:
                avg_neighbor_words = float(max_words_generation)
                std_neighbor_words = 0.0

            target_min_words = max(1, int(round(avg_neighbor_words - (2 * std_neighbor_words))))
            target_max_words = min(
                max_words_generation,
                int(round(avg_neighbor_words + (2 * std_neighbor_words))),
            )
            target_max_words = max(target_min_words, target_max_words)

            prompt = build_prompt_from_structured_with_neighbors(
                fields=generated.iloc[i].to_dict(),
                prompt_template=prompt_cfg["free_text_prompt"],
                max_words_generation=max_words_generation,
                text_column=text_column,
                neighbor_rows=sanitized_neighbor_rows,
                max_neighbors=int(llm_cfg["k_neighbors"]),
                max_words_reader=max_words_reader,
                target_min_words=target_min_words,
                target_max_words=target_max_words,
            )
            prompts.append(prompt)
        return prompts

    def _process_batch(
        batch_indices: list[int],
    ) -> tuple[dict[int, str], dict[int, str]]:
        prompts = _build_prompts_for_indices(batch_indices)
        row_updates: dict[int, str] = {}
        failed_updates: dict[int, str] = {}
        try:
            responses = llm_server.generate_batch(prompts, config=generation_config)
        except APIModelConnectionError as exc:
            # If it's a systematic API error (unsupported parameter/value, quota, etc.), stop immediately
            error_msg = str(exc).lower()
            if (
                "unsupported_parameter" in error_msg
                or "unsupported_value" in error_msg
                or "insufficient_quota" in error_msg
                or "quota" in error_msg
            ):
                print(f"\n❌ STOPPING: Systematic API error detected on batch:")
                print(f"   {exc}")
                raise
            # Otherwise treat as batch failure and continue with single-row fallback
            responses = []
        except Exception:
            responses = []

        if responses:
            for i, response_text in zip(batch_indices, responses):
                try:
                    cleaned_response = clean_generated_text(response_text)
                    if looks_like_meta_output(cleaned_response):
                        cleaned_response = ""
                    row_updates[i] = cap_words(
                        cleaned_response,
                        max_words_generation,
                    )
                except Exception as exc:
                    failed_updates[i] = f"{type(exc).__name__}: {exc}"

        if not responses or len(responses) != len(batch_indices):
            pending_indices = [
                i for i in batch_indices if i not in row_updates and i not in failed_updates
            ]
            for i in pending_indices:
                text, error = _process_single_row(i)
                if error is None and text is not None:
                    row_updates[i] = text
                else:
                    failed_updates[i] = error or "No response was returned for this row."
        missing_indices = set(batch_indices) - set(row_updates) - set(failed_updates)
        for i in sorted(missing_indices):
            failed_updates[i] = "No response was returned for this row."
        return row_updates, failed_updates

    def _process_single_row(i: int) -> tuple[str | None, str | None]:
        try:
            prompt = _build_prompts_for_indices([i])[0]
            response_text = llm_server.generate_batch([prompt], config=generation_config)[0]
            cleaned_response = clean_generated_text(response_text)
            if looks_like_meta_output(cleaned_response):
                cleaned_response = ""
            return cap_words(cleaned_response, max_words_generation), None
        except APIModelConnectionError as exc:
            # If it's a systematic API error (unsupported parameter/value, quota, etc.), stop immediately
            error_msg = str(exc).lower()
            if (
                "unsupported_parameter" in error_msg
                or "unsupported_value" in error_msg
                or "insufficient_quota" in error_msg
                or "quota" in error_msg
            ):
                print(f"\n❌ STOPPING: Systematic API error detected:")
                print(f"   {exc}")
                print(f"\n   Row {i} encountered a model compatibility issue.")
                print(f"   This error affects ALL rows - stopping pipeline immediately.")
                raise
            # Otherwise return error message for retry logic
            return None, f"APIModelConnectionError: {exc}"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _run_sequential() -> tuple[pd.DataFrame, dict[int, str]]:
        failed_rows: dict[int, str] = {}
        progress = tqdm(
            range(0, total_rows, batch_size),
            desc=f"Generating {text_column}",
            dynamic_ncols=True,
        )

        for batch_num, batch_start in enumerate(progress, start=1):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_indices = list(range(batch_start, batch_end))
            row_updates, batch_failures = _process_batch(batch_indices)

            if batch_log_every and batch_num % batch_log_every == 0:
                progress.write(
                    "  • Batch "
                    f"{batch_num}: rows [{batch_start}:{batch_end}] "
                    f"-> received {len(batch_indices)} response(s)."
                )

            for i, text in row_updates.items():
                generated.at[generated.index[i], text_column] = text
            failed_rows.update(batch_failures)

            time.sleep(sleep_sec)

        return generated, failed_rows

    def _run_parallel_api() -> tuple[pd.DataFrame, dict[int, str]]:
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

        def _run_shard(
            shard_id: int,
            indices: list[int],
        ) -> tuple[int, dict[int, str], dict[int, str]]:
            shard_updates: dict[int, str] = {}
            shard_failures: dict[int, str] = {}
            for batch_num, offset in enumerate(range(0, len(indices), batch_size), start=1):
                batch_indices = indices[offset : offset + batch_size]
                row_updates, batch_failures = _process_batch(batch_indices)
                shard_updates.update(row_updates)
                shard_failures.update(batch_failures)
                progress_bars[shard_id].update(len(batch_indices))
                if batch_log_every and batch_num % batch_log_every == 0:
                    progress_bars[shard_id].write(
                        f"  • Shard {shard_id + 1} batch {batch_num}: "
                        f"generated {len(batch_indices)} row(s)."
                    )
                time.sleep(sleep_sec)
            return shard_id, shard_updates, shard_failures

        failed_rows: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(shard_indices)) as executor:
            futures = [
                executor.submit(_run_shard, shard_id, indices)
                for shard_id, indices in enumerate(shard_indices)
            ]
            for future in as_completed(futures):
                try:
                    _, shard_updates, shard_failures = future.result()
                    for i, text in shard_updates.items():
                        generated.at[generated.index[i], text_column] = text
                    failed_rows.update(shard_failures)
                except APIModelConnectionError as exc:
                    # Immediately stop on systematic API errors
                    error_msg = str(exc).lower()
                    if "unsupported_parameter" in error_msg or "unsupported_value" in error_msg:
                        print(f"\n❌ STOPPING: Systematic API error detected in parallel shard:")
                        print(f"   {exc}")
                        executor.shutdown(wait=False)
                        raise
                    raise

        for bar in progress_bars:
            bar.close()

        return generated, failed_rows

    def _retry_failed_rows(failed_rows: dict[int, str]) -> dict[int, str]:
        remaining_failures = dict(failed_rows)
        if not remaining_failures or not rerun_failed_rows or failed_row_retry_passes <= 0:
            return remaining_failures

        for retry_pass in range(1, failed_row_retry_passes + 1):
            retry_indices = sorted(remaining_failures)
            if not retry_indices:
                break

            print(
                f"- Retry pass {retry_pass}/{failed_row_retry_passes} "
                f"for {len(retry_indices)} failed row(s)."
            )
            next_failures: dict[int, str] = {}
            progress = tqdm(
                retry_indices,
                desc=f"Retrying {text_column}",
                dynamic_ncols=True,
            )
            for i in progress:
                text, error = _process_single_row(i)
                if error is None and text is not None:
                    generated.at[generated.index[i], text_column] = text
                else:
                    next_failures[i] = error or "No response was returned for this row."
                time.sleep(sleep_sec)

            recovered = len(remaining_failures) - len(next_failures)
            print(
                f"  • Retry pass {retry_pass} recovered {recovered} row(s); "
                f"{len(next_failures)} still failing."
            )
            remaining_failures = next_failures

        return remaining_failures

    if llm_server.provider in {"gemini", "openai"} and parallel_api_calls:
        generated, failed_rows = _run_parallel_api()
    else:
        generated, failed_rows = _run_sequential()

    failed_rows = _retry_failed_rows(failed_rows)
    if failed_rows:
        print(
            f"⚠️  Free-text generation left {len(failed_rows)} row(s) empty "
            "after retries/reruns."
        )
        sample_indices = list(sorted(failed_rows))[:5]
        for i in sample_indices:
            print(f"   - row {i}: {failed_rows[i]}")
    else:
        print("- All rows generated successfully after retries/reruns.")

    print("- Free-text generation complete.")

    return generated
