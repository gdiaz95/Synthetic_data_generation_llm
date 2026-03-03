from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from source.tabular.npgc import NPGC_special


@dataclass
class TabularGenerationResult:
    synthetic_data: pd.DataFrame
    z_correlated: pd.DataFrame
    real_z: pd.DataFrame
    nn_idx: np.ndarray


def knn_retrieval_step1(
    real_z: pd.DataFrame,
    syn_z: pd.DataFrame,
    k: int,
    cols: list[str] | None = None,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """For each synthetic row, find indices of the k nearest real rows in Z-space."""
    if not isinstance(real_z, pd.DataFrame) or not isinstance(syn_z, pd.DataFrame):
        raise TypeError("real_z and syn_z must be pandas DataFrames.")

    if cols is None:
        cols = list(real_z.columns.intersection(syn_z.columns))
        if not cols:
            raise ValueError("No overlapping columns between real_z and syn_z.")

    Xr = real_z[cols].to_numpy(dtype=np.float64, copy=False)
    Xs = syn_z[cols].to_numpy(dtype=np.float64, copy=False)

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


def run_tabular_generation(real_df: pd.DataFrame, tabular_cfg: dict) -> TabularGenerationResult:
    """Fit NPGC and generate synthetic tabular rows with neighbor indices."""
    tabular_features = real_df.drop(columns=["job_description"])

    synthesizer = NPGC_special(
        enforce_min_max_values=bool(tabular_cfg["enforce_min_max_values"]),
        epsilon=tabular_cfg["epsilon"],
    )
    synthesizer.fit(tabular_features)

    real_z = synthesizer._model_state["transformed_data"]
    synthetic_data, z_correlated = synthesizer.sample(len(tabular_features))

    nn_idx, _ = knn_retrieval_step1(
        real_z,
        z_correlated,
        k=int(tabular_cfg["knn_k"]),
    )

    return TabularGenerationResult(
        synthetic_data=synthetic_data,
        z_correlated=z_correlated,
        real_z=real_z,
        nn_idx=nn_idx,
    )
