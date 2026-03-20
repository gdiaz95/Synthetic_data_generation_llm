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
    frechet_real_train_vs_synthetic: float
    mmd_rbf_real_train_vs_synthetic: float
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



def _mean_rbf_kernel_blockwise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    gamma: float,
    batch_size: int,
    show_progress: bool = False,
) -> float:
    """Compute mean RBF kernel value in blocks to avoid O(n*m*d) allocation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    total = 0.0
    count = 0
    x_norm = np.sum(x * x, axis=1)
    y_norm = np.sum(y * y, axis=1)

    x_batch_starts = list(range(0, x.shape[0], batch_size))
    y_batch_starts = list(range(0, y.shape[0], batch_size))
    total_batches = len(x_batch_starts) * len(y_batch_starts)
    completed_batches = 0

    for i in x_batch_starts:
        xb = x[i : i + batch_size]
        xb_norm = x_norm[i : i + batch_size]
        for j in y_batch_starts:
            yb = y[j : j + batch_size]
            yb_norm = y_norm[j : j + batch_size]

            # Squared euclidean distances via ||a||^2 + ||b||^2 - 2a·b
            dist2 = xb_norm[:, None] + yb_norm[None, :] - 2.0 * (xb @ yb.T)
            np.maximum(dist2, 0.0, out=dist2)
            kernel = np.exp(-gamma * dist2)
            total += float(kernel.sum())
            count += int(kernel.size)
            completed_batches += 1

    if show_progress:
        print(f"[free-text] Completed MMD kernel blocks: {completed_batches}/{total_batches}")

    if count == 0:
        return float("nan")
    return float(total / count)



def _rbf_mmd_biased_batched(
    x: np.ndarray,
    y: np.ndarray,
    *,
    gamma: float | None = None,
    batch_size: int = 2048,
    show_progress: bool = False,
    progress_prefix: str = "",
) -> float:
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

    mean_kxx = _mean_rbf_kernel_blockwise(
        x,
        x,
        gamma=gamma,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    mean_kyy = _mean_rbf_kernel_blockwise(
        y,
        y,
        gamma=gamma,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    mean_kxy = _mean_rbf_kernel_blockwise(
        x,
        y,
        gamma=gamma,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    return float(mean_kxx + mean_kyy - 2.0 * mean_kxy)



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
    min_df: int = 2,
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

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=min_df)
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



def _pair_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mmd_batch_size: int = 2048,
    show_progress: bool = False,
    progress_prefix: str = "",
) -> tuple[float, float]:
    if show_progress:
        label = progress_prefix or "current pair"
        print(f"[free-text] Generating future distance for {label}...")
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)
    frechet = _frechet_distance(mu_x, cov_x, mu_y, cov_y)

    if show_progress:
        print(f"[free-text] Generating MMD for {label}...")
    mmd = _rbf_mmd_biased_batched(
        x,
        y,
        batch_size=mmd_batch_size,
        show_progress=show_progress,
        progress_prefix=progress_prefix,
    )
    return float(frechet), float(mmd)



def _build_structured_feature_matrices(
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    structured_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not structured_columns:
        empty_train = np.zeros((len(real_train_df), 0), dtype=np.float64)
        empty_holdout = np.zeros((len(real_holdout_df), 0), dtype=np.float64)
        empty_syn = np.zeros((len(synthetic_df), 0), dtype=np.float64)
        return empty_train, empty_holdout, empty_syn

    missing_real_train = [col for col in structured_columns if col not in real_train_df.columns]
    missing_real_holdout = [col for col in structured_columns if col not in real_holdout_df.columns]
    missing_syn = [col for col in structured_columns if col not in synthetic_df.columns]
    if missing_real_train or missing_real_holdout or missing_syn:
        raise ValueError(
            "Structured columns missing in one or more dataframes. "
            f"train_missing={missing_real_train}, holdout_missing={missing_real_holdout}, synthetic_missing={missing_syn}"
        )

    combined = pd.concat(
        [
            real_train_df[structured_columns],
            real_holdout_df[structured_columns],
            synthetic_df[structured_columns],
        ],
        axis=0,
        ignore_index=True,
    )
    lengths = [len(real_train_df), len(real_holdout_df), len(synthetic_df)]

    blocks: list[np.ndarray] = []
    for column_name in structured_columns:
        series = combined[column_name]
        if is_numeric_dtype(series) or series.dropna().map(lambda value: isinstance(value, bool)).all():
            numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float64)
            blocks.append(numeric.to_numpy(dtype=np.float64).reshape(-1, 1))
            continue

        categorical = pd.get_dummies(
            series.fillna("__missing__").astype(str),
            prefix=column_name,
            dtype=np.float64,
        )
        blocks.append(categorical.to_numpy(dtype=np.float64, copy=False))

    feature_matrix = np.hstack(blocks) if blocks else np.zeros((len(combined), 0), dtype=np.float64)

    matrices: list[np.ndarray] = []
    offset = 0
    for length in lengths:
        matrices.append(feature_matrix[offset : offset + length])
        offset += length

    return tuple(matrices)



def _build_text_feature_matrices(
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    text_columns: list[str],
    embedding_dim: int,
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not text_columns:
        empty_train = np.zeros((len(real_train_df), 0), dtype=np.float64)
        empty_holdout = np.zeros((len(real_holdout_df), 0), dtype=np.float64)
        empty_syn = np.zeros((len(synthetic_df), 0), dtype=np.float64)
        return empty_train, empty_holdout, empty_syn

    train_blocks: list[np.ndarray] = []
    holdout_blocks: list[np.ndarray] = []
    synthetic_blocks: list[np.ndarray] = []

    for column_name in text_columns:
        if column_name not in real_train_df.columns:
            raise ValueError(f"Free-text column '{column_name}' is missing in real train data.")
        if column_name not in real_holdout_df.columns:
            raise ValueError(f"Free-text column '{column_name}' is missing in real holdout data.")
        if column_name not in synthetic_df.columns:
            raise ValueError(f"Free-text column '{column_name}' is missing in synthetic data.")

        embedded_sets, _ = _embed_tfidf_svd_many(
            [
                _clean_text_series_preserve_rows(real_train_df[column_name]),
                _clean_text_series_preserve_rows(real_holdout_df[column_name]),
                _clean_text_series_preserve_rows(synthetic_df[column_name]),
            ],
            embedding_dim=embedding_dim,
            max_features=max_features,
        )
        train_emb, holdout_emb, synthetic_emb = embedded_sets
        train_blocks.append(_row_normalize(train_emb))
        holdout_blocks.append(_row_normalize(holdout_emb))
        synthetic_blocks.append(_row_normalize(synthetic_emb))

    train_matrix = np.hstack(train_blocks) if train_blocks else np.zeros((len(real_train_df), 0), dtype=np.float64)
    holdout_matrix = np.hstack(holdout_blocks) if holdout_blocks else np.zeros((len(real_holdout_df), 0), dtype=np.float64)
    synthetic_matrix = np.hstack(synthetic_blocks) if synthetic_blocks else np.zeros((len(synthetic_df), 0), dtype=np.float64)

    return _row_normalize(train_matrix), _row_normalize(holdout_matrix), _row_normalize(synthetic_matrix)



def _holdout_discriminator_metrics(
    reference_x: np.ndarray,
    holdout_x: np.ndarray,
    *,
    random_state: int,
) -> tuple[float, float]:
    if reference_x.shape[1] != holdout_x.shape[1]:
        raise ValueError("Reference and holdout feature matrices must have the same width.")
    if len(reference_x) < 2 or len(holdout_x) < 2:
        raise ValueError("Need at least two rows from each class for discriminator evaluation.")

    x = np.vstack([reference_x, holdout_x])
    y = np.concatenate(
        [
            np.zeros(len(reference_x), dtype=np.int64),
            np.ones(len(holdout_x), dtype=np.int64),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)

    return float(roc_auc_score(y_test, probabilities)), float(accuracy_score(y_test, predictions))



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
    mmd_batch_size: int = 2048,
    show_batch_progress: bool = False,
    progress_prefix: str = "",
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

    if show_batch_progress:
        print(f"[free-text] Generating embeddings for {progress_prefix or text_column}...")
    embedded_sets, latent_dim = _embed_tfidf_svd_many(
        [real_train_text, real_holdout_text, syn_text],
        embedding_dim=embedding_dim,
        max_features=max_features,
    )
    real_train_emb, real_holdout_emb, syn_emb = embedded_sets

    frechet_train_syn, mmd_train_syn = _pair_metrics(
        real_train_emb,
        syn_emb,
        mmd_batch_size=mmd_batch_size,
        show_progress=show_batch_progress,
        progress_prefix=f"{progress_prefix} train-vs-syn".strip(),
    )
    frechet_train_holdout, mmd_train_holdout = _pair_metrics(
        real_train_emb,
        real_holdout_emb,
        mmd_batch_size=mmd_batch_size,
        show_progress=show_batch_progress,
        progress_prefix=f"{progress_prefix} train-vs-holdout".strip(),
    )

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
            frechet_real_train_vs_synthetic=float(frechet_train_syn),
            mmd_rbf_real_train_vs_synthetic=float(mmd_train_syn),
            frechet_real_train_vs_real_holdout=float(frechet_train_holdout),
            mmd_rbf_real_train_vs_real_holdout=float(mmd_train_holdout),
            frechet_ratio_to_holdout=_safe_ratio(frechet_train_syn, frechet_train_holdout),
            mmd_ratio_to_holdout=_safe_ratio(mmd_train_syn, mmd_train_holdout),
        )
    )

