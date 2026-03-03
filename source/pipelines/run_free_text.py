from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from source.llm.server_model import SingleGPUServer


def cap_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", (text or "").strip())
    return " ".join(words[:max_words]).strip()


def build_prompt_from_structured_with_neighbors(
    fields: dict[str, Any],
    prompt_template: str,
    max_words: int,
    neighbor_texts: list[str] | None = None,
    max_neighbors: int = 3,
    neighbor_excerpt_words: int = 20,
) -> str:
    clean = {
        key: value
        for key, value in fields.items()
        if key != "job_description" and value is not None and pd.notna(value)
    }
    

    snippets: list[str] = []
    if neighbor_texts:
        for text in neighbor_texts[:max_neighbors]:
            excerpt = cap_words(str(text), neighbor_excerpt_words)
            if excerpt:
                snippets.append(f"- {excerpt}")

    neighbor_block = "\n".join(snippets) if snippets else "(none)"

    return prompt_template.format(
        max_words=max_words,
        structured_fields=clean,
        neighbor_block=neighbor_block,
    ).strip()


def generate_job_descriptions(
    llm_server: SingleGPUServer,
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    nn_idx,
    llm_cfg: dict,
    prompt_cfg: dict,
) -> pd.DataFrame:
    generated = synthetic_df.copy()
    generated["job_description"] = ""

    for i in tqdm(range(len(generated)), desc="Generating job descriptions"):
        neighbor_pos = nn_idx[i, : int(llm_cfg["k_neighbors"])]
        neighbor_texts = (
            original_df.iloc[neighbor_pos]["job_description"]
            .dropna()
            .astype(str)
            .tolist()
        )

        prompt = build_prompt_from_structured_with_neighbors(
            fields=generated.iloc[i].to_dict(),
            prompt_template=prompt_cfg["job_description_prompt"],
            max_words=int(llm_cfg["max_words"]),
            neighbor_texts=neighbor_texts,
            max_neighbors=int(llm_cfg["k_neighbors"]),
            neighbor_excerpt_words=int(prompt_cfg["neighbor_excerpt_words"]),
        )

        response_text = llm_server.generate_batch(
            [prompt],
            config={
                "temperature": float(llm_cfg["temperature"]),
                "top_p": float(llm_cfg["top_p"]),
                "max_tokens": int(llm_cfg["max_tokens"]),
            },
        )[0]

        generated.at[generated.index[i], "job_description"] = cap_words(
            response_text,
            int(llm_cfg["max_words"]),
        )

        time.sleep(float(llm_cfg["sleep_sec"]))

    return generated
