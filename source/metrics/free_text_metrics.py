from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class FreeTextDistributionSummary:
    """Distribution metrics for a single free-text column.

    Calibration strategy:
    - Compute real-train vs synthetic (target gap).
    - Compute real-train vs real-holdout (natural baseline gap).
    - Report ratios so the synthetic gap is interpretable.

    Efficiency:
    - Build one shared embedding space over all required text sets once.
    - Reuse the resulting embeddings for every pairwise distance.
    """

    text_column: str
    embedding_backend: str
    embedding_dim: int
    n_real_train: int
    n_real_holdout: int
    n_synthetic: int
    frechet_distance: float
    mmd_rbf: float
    frechet_real_train_vs_real_holdout: float
    mmd_rbf_real_train_vs_real_holdout: float
    frechet_ratio_to_holdout: float
    mmd_ratio_to_holdout: float


def _clean_text_series(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text[text != ""]


def _rbf_mmd_biased(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")

    if gamma is None:
        # Median heuristic on a small shared subset for stable runtime.
        sample_x = x[: min(500, len(x))]
        sample_y = y[: min(500, len(y))]
        pair = np.vstack([sample_x, sample_y])
        dist2 = np.sum((pair[:, None, :] - pair[None, :, :]) ** 2, axis=2)
        median = float(np.median(dist2[dist2 > 0])) if np.any(dist2 > 0) else 1.0
        gamma = 1.0 / max(median, 1e-9)

    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2))
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def _frechet_distance(
    mu_real: np.ndarray,
    cov_real: np.ndarray,
    mu_syn: np.ndarray,
    cov_syn: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    cov_real = cov_real + np.eye(cov_real.shape[0]) * eps
    cov_syn = cov_syn + np.eye(cov_syn.shape[0]) * eps

    covmean, _ = linalg.sqrtm(cov_real @ cov_syn, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    mean_diff = mu_real - mu_syn
    return float(mean_diff @ mean_diff + np.trace(cov_real + cov_syn - 2.0 * covmean))


def _embed_tfidf_svd_many(
    text_sets: list[pd.Series],
    *,
    embedding_dim: int,
    max_features: int,
) -> tuple[list[np.ndarray], int]:
    cleaned_sets = [list(series) for series in text_sets]
    lengths = [len(items) for items in cleaned_sets]
    if any(length == 0 for length in lengths):
        raise ValueError("All text sets must contain at least one non-empty row.")

    joined: list[str] = []
    for items in cleaned_sets:
        joined.extend(items)

    if len(joined) < 2:
        raise ValueError("Need at least two total text rows to compute distribution metrics.")

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    tfidf = vectorizer.fit_transform(joined)

    # Keep target dimensionality high by default; cap by matrix rank.
    max_rank = max(1, min(tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    latent_dim = max(1, min(int(embedding_dim), max_rank))
    svd = TruncatedSVD(n_components=latent_dim, random_state=42)
    emb = svd.fit_transform(tfidf).astype(np.float64, copy=False)

    split_embeddings: list[np.ndarray] = []
    offset = 0
    for length in lengths:
        split_embeddings.append(emb[offset : offset + length])
        offset += length

    return split_embeddings, latent_dim


def _pair_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)
    frechet = _frechet_distance(mu_x, cov_x, mu_y, cov_y)
    mmd = _rbf_mmd_biased(x, y)
    return float(frechet), float(mmd)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def evaluate_first_free_text_distribution(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    free_text_columns: list[str],
    *,
    real_holdout_df: pd.DataFrame | None = None,
    embedding_dim: int = 768,
    max_features: int = 200_000,
) -> dict[str, float | int | str]:
    """Evaluate distribution matching for the first configured free-text column.

    If `real_holdout_df` is provided, metrics are calibrated against a real-vs-real
    baseline in the exact same embedding space without recomputing embeddings.
    """

    if not free_text_columns:
        raise ValueError("No free-text columns provided for evaluation.")

    text_column = free_text_columns[0]
    if text_column not in real_df.columns:
        raise ValueError(f"Free-text column '{text_column}' is missing in real data.")
    if text_column not in synthetic_df.columns:
        raise ValueError(f"Free-text column '{text_column}' is missing in synthetic data.")

    if real_holdout_df is None:
        real_train_text = _clean_text_series(real_df[text_column])
        real_holdout_text = real_train_text
    else:
        if text_column not in real_holdout_df.columns:
            raise ValueError(f"Free-text column '{text_column}' is missing in real holdout data.")
        real_train_text = _clean_text_series(real_df[text_column])
        real_holdout_text = _clean_text_series(real_holdout_df[text_column])

    syn_text = _clean_text_series(synthetic_df[text_column])

    if real_train_text.empty or real_holdout_text.empty or syn_text.empty:
        raise ValueError("Free-text metric requires non-empty real-train, real-holdout, and synthetic text rows.")

    embedded_sets, latent_dim = _embed_tfidf_svd_many(
        [real_train_text, real_holdout_text, syn_text],
        embedding_dim=embedding_dim,
        max_features=max_features,
    )
    real_train_emb, real_holdout_emb, syn_emb = embedded_sets

    frechet_train_syn, mmd_train_syn = _pair_metrics(real_train_emb, syn_emb)
    frechet_train_holdout, mmd_train_holdout = _pair_metrics(real_train_emb, real_holdout_emb)

    return asdict(
        FreeTextDistributionSummary(
            text_column=text_column,
            embedding_backend="tfidf_svd",
            embedding_dim=int(latent_dim),
            n_real_train=int(real_train_emb.shape[0]),
            n_real_holdout=int(real_holdout_emb.shape[0]),
            n_synthetic=int(syn_emb.shape[0]),
            frechet_distance=float(frechet_train_syn),
            mmd_rbf=float(mmd_train_syn),
            frechet_real_train_vs_real_holdout=float(frechet_train_holdout),
            mmd_rbf_real_train_vs_real_holdout=float(mmd_train_holdout),
            frechet_ratio_to_holdout=_safe_ratio(frechet_train_syn, frechet_train_holdout),
            mmd_ratio_to_holdout=_safe_ratio(mmd_train_syn, mmd_train_holdout),
        )
    )
