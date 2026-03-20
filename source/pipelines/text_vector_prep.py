from __future__ import annotations

import hashlib
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


def prepare_text_vector_features(series: pd.Series, *, hash_dim: int = 8) -> pd.DataFrame:
    """Build a row-normalized hashed text vector block with exactly ``hash_dim`` features."""
    if hash_dim <= 0:
        raise ValueError("hash_dim must be >= 1.")

    rows: list[np.ndarray] = []
    for value in series:
        text = _safe_text(value)
        tokens = _tokenize(text)
        hashed = _hash_embedding(tokens, hash_dim)
        rows.append(hashed)

    if not rows:
        return pd.DataFrame()

    matrix = np.vstack(rows)
    columns = [f"hash_{i}" for i in range(hash_dim)]
    return pd.DataFrame(matrix, index=series.index, columns=columns)
