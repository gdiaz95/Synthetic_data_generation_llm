from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def _hash_embedding(tokens: list[str], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    if not tokens:
        return vec
    for token in tokens:
        idx = _hash_token(token, dim)
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0.0:
        vec /= norm
    return vec


def _scalar_features(text: str, tokens: list[str]) -> np.ndarray:
    char_count = float(len(text))
    word_count = float(len(tokens))
    unique_ratio = float(len(set(tokens)) / len(tokens)) if tokens else 0.0
    punctuation_ratio = float(sum(1 for ch in text if not ch.isalnum() and not ch.isspace()) / max(1, len(text)))
    avg_token_len = float(sum(len(tok) for tok in tokens) / len(tokens)) if tokens else 0.0
    return np.array(
        [
            math.log1p(char_count),
            math.log1p(word_count),
            unique_ratio,
            punctuation_ratio,
            avg_token_len,
        ],
        dtype=np.float64,
    )


def prepare_text_vector_features(series: pd.Series, *, hash_dim: int = 8) -> pd.DataFrame:
    """Build a lightweight hybrid text vector block: hash embedding + scalar stats.

    This is intentionally simple and fast for pipeline staging; distance integration
    can consume this block in a later iteration.
    """
    if hash_dim <= 0:
        raise ValueError("hash_dim must be >= 1.")

    rows: list[np.ndarray] = []
    for value in series:
        text = _safe_text(value)
        tokens = _tokenize(text)
        hashed = _hash_embedding(tokens, hash_dim)
        stats = _scalar_features(text, tokens)
        rows.append(np.concatenate([hashed, stats]))

    if not rows:
        return pd.DataFrame()

    matrix = np.vstack(rows)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0.0] = 1.0
    standardized = (matrix - means) / stds

    columns = [f"hash_{i}" for i in range(hash_dim)] + [
        "stat_log_chars",
        "stat_log_words",
        "stat_lexical_diversity",
        "stat_punctuation_ratio",
        "stat_avg_token_len",
    ]
    return pd.DataFrame(standardized, index=series.index, columns=columns)
