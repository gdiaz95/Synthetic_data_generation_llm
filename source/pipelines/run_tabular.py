from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from source.tabular.npgc import NPGC_special


def _row_normalize_dense(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


@dataclass
class TabularGenerationResult:
    synthetic_data: pd.DataFrame
    z_correlated: pd.DataFrame
    real_z: pd.DataFrame
    nn_idx: np.ndarray | None


def knn_retrieval_step1(
    real_z: pd.DataFrame,
    syn_z: pd.DataFrame,
    k: int,
    cols: list[str] | None = None,
    feature_weights: dict[str, float] | None = None,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """For each synthetic row, find indices of the k nearest real rows in Z-space."""
    if not isinstance(real_z, pd.DataFrame) or not isinstance(syn_z, pd.DataFrame):
        raise TypeError("real_z and syn_z must be pandas DataFrames.")

    if cols is None:
        cols = list(real_z.columns.intersection(syn_z.columns))
        if not cols:
            raise ValueError("No overlapping columns between real_z and syn_z.")

    Xr = _row_normalize_dense(real_z[cols].to_numpy(dtype=np.float64, copy=False))
    Xs = _row_normalize_dense(syn_z[cols].to_numpy(dtype=np.float64, copy=False))

    if feature_weights:
        weights = np.array(
            [float(feature_weights.get(col, 1.0)) for col in cols],
            dtype=np.float64,
        )
    else:
        weights = np.ones(len(cols), dtype=np.float64)

    if np.any(weights < 0):
        raise ValueError("Feature weights must be non-negative.")

    scales = np.sqrt(weights)
    Xr = Xr * scales
    Xs = Xs * scales

    n = Xr.shape[0]
    m = Xs.shape[0]

    if k <= 0:
        raise ValueError("k must be >= 1.")
    if k > n:
        raise ValueError(f"k={k} cannot exceed number of real rows n={n}.")

    r_norm2 = np.sum(Xr * Xr, axis=1)
    nn_idx = np.empty((m, k), dtype=np.int64)
    nn_dist = np.empty((m, k), dtype=np.float64)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        S = Xs[start:end]
        s_norm2 = np.sum(S * S, axis=1)

        dist2 = s_norm2[:, None] + r_norm2[None, :] - 2.0 * (S @ Xr.T)
        np.maximum(dist2, 0.0, out=dist2)

        part = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        row_ids = np.arange(end - start)[:, None]
        part_dist2 = dist2[row_ids, part]
        order = np.argsort(part_dist2, axis=1)
        knn = part[row_ids, order]
        knn_dist = np.sqrt(part_dist2[row_ids, order])

        nn_idx[start:end] = knn
        nn_dist[start:end] = knn_dist

    return nn_idx, nn_dist


def knn_retrieval_with_text_blocks(
    real_z: pd.DataFrame,
    syn_z: pd.DataFrame,
    k: int,
    *,
    text_blocks: list[tuple[pd.DataFrame, pd.DataFrame, float]] | None = None,
    cols: list[str] | None = None,
    feature_weights: dict[str, float] | None = None,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """KNN retrieval using the tabular block plus optional text-vector distance blocks.

    Each block (tabular, and every text block) is L2 row-normalized to unit
    norm before distances are computed. Distance uses the form:
        D(i,j) = d_struct^2 + sum_c lambda_c * ||z_c(s) - z_c(r)||^2
    with d_struct^2 and each ||z_c(s) - z_c(r)||^2 computed directly on the
    unit-normalized vectors (no additional per-block dimension division).
    The structured block's coefficient is always 1 (not configurable); each
    text block carries its own lambda_c weight, passed via ``text_blocks``.
    """
    if cols is None:
        cols = list(real_z.columns.intersection(syn_z.columns))
        if not cols:
            raise ValueError("No overlapping columns between real_z and syn_z.")

    Xr_tab = _row_normalize_dense(real_z[cols].to_numpy(dtype=np.float64, copy=False))
    Xs_tab = _row_normalize_dense(syn_z[cols].to_numpy(dtype=np.float64, copy=False))

    if feature_weights:
        weights = np.array([float(feature_weights.get(col, 1.0)) for col in cols], dtype=np.float64)
    else:
        weights = np.ones(len(cols), dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("Feature weights must be non-negative.")

    tab_scales = np.sqrt(weights)
    Xr_tab = Xr_tab * tab_scales
    Xs_tab = Xs_tab * tab_scales

    n = Xr_tab.shape[0]
    m = Xs_tab.shape[0]
    if k <= 0:
        raise ValueError("k must be >= 1.")
    if k > n:
        raise ValueError(f"k={k} cannot exceed number of real rows n={n}.")

    r_tab_norm2 = np.sum(Xr_tab * Xr_tab, axis=1)

    prepared_text: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for real_block, syn_block, lam in text_blocks or []:
        if lam < 0:
            raise ValueError("All lambda (text block) weights must be non-negative.")
        block_cols = list(real_block.columns.intersection(syn_block.columns))
        if not block_cols:
            continue
        Xr = _row_normalize_dense(real_block[block_cols].to_numpy(dtype=np.float64, copy=False))
        Xs = _row_normalize_dense(syn_block[block_cols].to_numpy(dtype=np.float64, copy=False))
        if Xr.shape[0] != n or Xs.shape[0] != m:
            raise ValueError("Text block row counts must match tabular real/synthetic row counts.")
        r_norm2 = np.sum(Xr * Xr, axis=1)
        prepared_text.append((Xr, Xs, r_norm2, float(lam)))

    nn_idx = np.empty((m, k), dtype=np.int64)
    nn_dist = np.empty((m, k), dtype=np.float64)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        S_tab = Xs_tab[start:end]
        s_tab_norm2 = np.sum(S_tab * S_tab, axis=1)
        dist2 = s_tab_norm2[:, None] + r_tab_norm2[None, :] - 2.0 * (S_tab @ Xr_tab.T)

        for Xr, Xs, r_norm2, lam in prepared_text:
            S = Xs[start:end]
            s_norm2 = np.sum(S * S, axis=1)
            dist2 += lam * (s_norm2[:, None] + r_norm2[None, :] - 2.0 * (S @ Xr.T))

        np.maximum(dist2, 0.0, out=dist2)

        part = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        row_ids = np.arange(end - start)[:, None]
        part_dist2 = dist2[row_ids, part]
        order = np.argsort(part_dist2, axis=1)
        knn = part[row_ids, order]
        knn_dist = np.sqrt(part_dist2[row_ids, order])

        nn_idx[start:end] = knn
        nn_dist[start:end] = knn_dist

    return nn_idx, nn_dist


def run_tabular_generation(
    real_df: pd.DataFrame,
    tabular_cfg: dict,
    text_columns: list[str],
    feature_weights: dict[str, float] | None = None,
    compute_knn: bool = True,
    max_rows: int | None = None,
) -> TabularGenerationResult:
    """Fit NPGC and generate synthetic tabular rows with neighbor indices."""
    excluded = [col for col in text_columns if col in real_df.columns]
    print(f"- Preparing tabular features (excluding free-text columns: {excluded})...")
    tabular_features = real_df.drop(columns=excluded)
    print(f"- Tabular feature matrix shape: {tabular_features.shape}")

    print("- Initializing NPGC synthesizer...")
    synthesizer = NPGC_special(
        enforce_min_max_values=bool(tabular_cfg["enforce_min_max_values"]),
        epsilon=tabular_cfg["epsilon"],
    )
    print("- Fitting NPGC synthesizer on tabular features...")
    synthesizer.fit(tabular_features)

    real_z = synthesizer._model_state["transformed_data"]
    print("- Sampling synthetic tabular rows...")
    n_samples = min(len(tabular_features), max_rows) if max_rows else len(tabular_features)
    synthetic_data, z_correlated = synthesizer.sample(n_samples)
    print(
        f"- Sampled synthetic rows: {synthetic_data.shape[0]} | "
        f"Columns: {synthetic_data.shape[1]}"
    )

    nn_idx: np.ndarray | None = None
    if text_columns and compute_knn:
        print(f"- Running KNN retrieval in latent space with k={int(tabular_cfg['knn_k'])}...")
        nn_idx, _ = knn_retrieval_step1(
            real_z,
            z_correlated,
            k=int(tabular_cfg["knn_k"]),
            feature_weights=feature_weights,
        )
        print("- Tabular generation and KNN retrieval complete.")
    elif text_columns:
        print("- Skipping KNN retrieval; tabular-only generation requested.")
    else:
        print("- No free-text columns configured; skipping KNN retrieval.")

    return TabularGenerationResult(
        synthetic_data=synthetic_data,
        z_correlated=z_correlated,
        real_z=real_z,
        nn_idx=nn_idx,
    )
