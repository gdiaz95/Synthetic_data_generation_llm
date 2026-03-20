from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from source.metrics.free_text_metrics import _embed_tfidf_svd_many
from source.tabular.npgc import NPGC_special


@dataclass
class QAMetricsSummary:
    overall_accuracy: float
    univariate_accuracy: float
    bivariate_accuracy: float
    discriminator_auc: float
    identical_matches: float
    dcr_training: float
    dcr_holdout: float
    dcr_share: float


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _looks_like_metrics_object(obj: Any) -> bool:
    return (
        obj is not None
        and hasattr(obj, "accuracy")
        and hasattr(obj, "similarity")
        and hasattr(obj, "distances")
    )


def _extract_metrics(report_result: Any) -> Any:
    """Handle return-shape differences across mostlyai qa.report versions."""
    if _looks_like_metrics_object(report_result):
        return report_result

    if isinstance(report_result, tuple):
        for item in report_result:
            if _looks_like_metrics_object(item):
                return item
            nested = getattr(item, "metrics", None)
            if _looks_like_metrics_object(nested):
                return nested

    nested = getattr(report_result, "metrics", None)
    if _looks_like_metrics_object(nested):
        return nested

    raise ValueError(
        "Could not extract metrics object from mostlyai.qa.report output. "
        f"Received type={type(report_result)}."
    )


def _to_summary(metrics: Any) -> QAMetricsSummary:
    """Convert mostlyai qa metrics object into a lightweight serializable dataclass."""
    return QAMetricsSummary(
        overall_accuracy=_safe_float(metrics.accuracy.overall),
        univariate_accuracy=_safe_float(metrics.accuracy.univariate),
        bivariate_accuracy=_safe_float(metrics.accuracy.bivariate),
        discriminator_auc=_safe_float(metrics.similarity.discriminator_auc_training_synthetic),
        identical_matches=_safe_float(metrics.distances.ims_training),
        dcr_training=_safe_float(metrics.distances.dcr_training),
        dcr_holdout=_safe_float(metrics.distances.dcr_holdout),
        dcr_share=_safe_float(metrics.distances.dcr_share),
    )


def evaluate_structured_qa_metrics(
    *,
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    non_free_text_columns: list[str],
) -> dict[str, float]:
    """
    Evaluate MOSTLY AI QA metrics using only non-free-text (structured) columns.

    The function intentionally ignores free-text columns for now.
    """
    try:
        from mostlyai import qa
    except ImportError as exc:
        raise ImportError(
            "mostlyai is required to compute QA metrics. Install it in your environment first."
        ) from exc

    if not non_free_text_columns:
        raise ValueError("No structured columns were provided for evaluation.")

    missing_train = [c for c in non_free_text_columns if c not in train_df.columns]
    missing_syn = [c for c in non_free_text_columns if c not in synthetic_df.columns]
    missing_holdout = [c for c in non_free_text_columns if c not in holdout_df.columns]
    if missing_train or missing_syn or missing_holdout:
        raise ValueError(
            "Structured columns missing in one or more dataframes. "
            f"train_missing={missing_train}, synthetic_missing={missing_syn}, holdout_missing={missing_holdout}"
        )

    train_eval = train_df[non_free_text_columns]
    synthetic_eval = synthetic_df[non_free_text_columns]
    holdout_eval = holdout_df[non_free_text_columns]

    report_result = qa.report(
        syn_tgt_data=synthetic_eval,
        trn_tgt_data=train_eval,
        hol_tgt_data=holdout_eval,
    )
    metrics = _extract_metrics(report_result)

    report_file = Path("model-report.html")
    if report_file.exists():
        report_file.unlink()

    return asdict(_to_summary(metrics))


def _infer_prediction_task(target: pd.Series) -> str:
    non_null = target.dropna()
    if non_null.empty:
        raise ValueError("Target column has no non-null rows for TSTR.")

    unique_count = int(non_null.nunique(dropna=True))
    if not pd.api.types.is_numeric_dtype(non_null):
        return "classification"
    if unique_count <= min(20, max(2, int(0.1 * len(non_null)))):
        return "classification"
    return "regression"


def _normalize_block(block: Any) -> Any:
    if sparse.issparse(block):
        return normalize(block, norm="l2", axis=1, copy=False)
    return normalize(block, norm="l2", axis=1)


def _to_sparse_block(block: Any) -> sparse.csr_matrix:
    if sparse.issparse(block):
        return block.tocsr()
    return sparse.csr_matrix(block)


def _build_structured_block(
    reference_features: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    structured_columns: list[str],
) -> tuple[Any, Any, int]:
    if not structured_columns:
        empty_train = sparse.csr_matrix((len(train_features), 0))
        empty_test = sparse.csr_matrix((len(test_features), 0))
        return empty_train, empty_test, 0

    reference_block = reference_features[structured_columns].copy()
    for column_name in structured_columns:
        if reference_block[column_name].isna().all():
            reference_block[column_name] = 0
        elif pd.api.types.is_numeric_dtype(reference_block[column_name]):
            fill_value = float(reference_block[column_name].median())
            reference_block[column_name] = reference_block[column_name].fillna(fill_value)
        else:
            mode = reference_block[column_name].mode(dropna=True)
            fill_value = "" if mode.empty else mode.iloc[0]
            reference_block[column_name] = reference_block[column_name].fillna(fill_value)

    synthesizer = NPGC_special(epsilon=None)
    synthesizer.fit(reference_block)
    train_block = synthesizer.transform(train_features[structured_columns]).to_numpy(dtype=float)
    test_block = synthesizer.transform(test_features[structured_columns]).to_numpy(dtype=float)
    return train_block, test_block, int(len(structured_columns))


def _build_text_block(
    train_text: pd.Series,
    test_text: pd.Series,
    *,
    embedding_dim: int,
    max_features: int,
) -> tuple[Any, Any, int]:
    clean_train = train_text.fillna("").astype(str).str.strip()
    clean_test = test_text.fillna("").astype(str).str.strip()
    try:
        embeddings, latent_dim = _embed_tfidf_svd_many(
            [clean_train, clean_test],
            embedding_dim=embedding_dim,
            max_features=max_features,
            min_df=1,
        )
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        empty_train = sparse.csr_matrix((len(train_text), 0))
        empty_test = sparse.csr_matrix((len(test_text), 0))
        return empty_train, empty_test, 0
    train_matrix, test_matrix = embeddings
    return train_matrix, test_matrix, int(latent_dim)


def _build_weighted_feature_matrices(
    *,
    reference_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    structured_columns: list[str],
    text_columns: list[str],
    structured_weight: float,
    text_weight_default: float,
    text_weight_by_column: dict[str, float] | None = None,
    text_embedding_dim: int = 768,
    text_max_features: int = 50_000,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    text_weight_by_column = text_weight_by_column or {}

    train_blocks: list[Any] = []
    test_blocks: list[Any] = []
    text_diagnostics: dict[str, dict[str, float | int]] = {}

    structured_train, structured_test, structured_width = _build_structured_block(
        reference_df,
        train_df,
        test_df,
        structured_columns,
    )
    if structured_width > 0:
        structured_train = _to_sparse_block(_normalize_block(structured_train) * float(structured_weight))
        structured_test = _to_sparse_block(_normalize_block(structured_test) * float(structured_weight))
        train_blocks.append(structured_train)
        test_blocks.append(structured_test)

    for column_name in text_columns:
        text_train, text_test, width = _build_text_block(
            train_df[column_name],
            test_df[column_name],
            embedding_dim=int(text_embedding_dim),
            max_features=int(text_max_features),
        )
        if width == 0:
            continue
        weight = float(text_weight_by_column.get(column_name, text_weight_default))
        text_train = _to_sparse_block(_normalize_block(text_train) * weight)
        text_test = _to_sparse_block(_normalize_block(text_test) * weight)
        train_blocks.append(text_train)
        test_blocks.append(text_test)
        text_diagnostics[column_name] = {
            "width": width,
            "weight": weight,
        }

    if not train_blocks:
        raise ValueError("Could not build any structured or text features.")

    x_train = sparse.hstack(train_blocks, format="csr")
    x_test = sparse.hstack(test_blocks, format="csr")
    diagnostics: dict[str, Any] = {
        "structured_block": {
            "width": structured_width,
            "weight": float(structured_weight),
        },
        "text_blocks": text_diagnostics,
        "total_feature_width": int(x_train.shape[1]),
    }
    return x_train, x_test, diagnostics


def _holdout_discriminator_metrics(
    train_x: sparse.csr_matrix,
    test_x: sparse.csr_matrix,
    *,
    random_state: int = 42,
) -> dict[str, float]:
    x = sparse.vstack([train_x, test_x], format="csr")
    y = np.concatenate(
        [
            np.zeros(train_x.shape[0], dtype=np.int64),
            np.ones(test_x.shape[0], dtype=np.int64),
        ]
    )
    x_train, x_eval, y_train, y_eval = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_eval)[:, 1]
    predictions = model.predict(x_eval)
    return {
        "roc_auc": float(roc_auc_score(y_eval, probabilities)),
        "accuracy": float(accuracy_score(y_eval, predictions)),
    }


def evaluate_tstr(
    *,
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_column: str,
    text_columns: list[str],
    structured_weight: float = 1.0,
    text_weight_default: float = 0.01,
    text_weight_by_column: dict[str, float] | None = None,
    text_embedding_dim: int = 768,
    text_max_features: int = 50_000,
) -> dict[str, Any]:
    text_weight_by_column = text_weight_by_column or {}

    for name, frame in {
        "train": train_df,
        "synthetic": synthetic_df,
        "holdout": holdout_df,
    }.items():
        if target_column not in frame.columns:
            raise ValueError(f"Target column '{target_column}' missing from {name} dataframe.")

    feature_columns = [col for col in synthetic_df.columns if col != target_column]
    available_text_columns = [col for col in text_columns if col in feature_columns]
    structured_columns = [col for col in feature_columns if col not in set(available_text_columns)]

    syn_eval = synthetic_df[feature_columns + [target_column]].dropna(subset=[target_column]).copy()
    holdout_eval = holdout_df[feature_columns + [target_column]].dropna(subset=[target_column]).copy()
    if syn_eval.empty or holdout_eval.empty:
        raise ValueError("Need non-empty synthetic and holdout rows with non-null target values for TSTR.")

    y_train = syn_eval[target_column]
    y_test = holdout_eval[target_column]
    task = _infer_prediction_task(y_train)

    x_train, x_test, feature_diagnostics = _build_weighted_feature_matrices(
        reference_df=train_df,
        train_df=syn_eval,
        test_df=holdout_eval,
        structured_columns=structured_columns,
        text_columns=available_text_columns,
        structured_weight=structured_weight,
        text_weight_default=text_weight_default,
        text_weight_by_column=text_weight_by_column,
        text_embedding_dim=text_embedding_dim,
        text_max_features=text_max_features,
    )

    metrics: dict[str, Any] = {
        "task": task,
        "target_column": target_column,
        "n_synthetic_train": int(len(syn_eval)),
        "n_holdout_test": int(len(holdout_eval)),
        "structured_columns": structured_columns,
        "text_columns": available_text_columns,
        "structured_block": feature_diagnostics["structured_block"],
        "text_blocks": feature_diagnostics["text_blocks"],
        "total_feature_width": feature_diagnostics["total_feature_width"],
    }

    if task == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train.astype(str))
        y_test_eval = y_test.astype(str)
        predictions = model.predict(x_test)
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_test_eval, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_eval, predictions)),
                "f1_weighted": float(f1_score(y_test_eval, predictions, average="weighted")),
            }
        )
        unique_labels = sorted(y_test_eval.unique())
        if len(unique_labels) == 2:
            probabilities = model.predict_proba(x_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test_eval, probabilities))
    else:
        model = Ridge()
        model.fit(x_train, pd.to_numeric(y_train, errors="coerce"))
        y_test_eval = pd.to_numeric(y_test, errors="coerce")
        predictions = model.predict(x_test)
        metrics.update(
            {
                "mae": float(mean_absolute_error(y_test_eval, predictions)),
                "r2": float(r2_score(y_test_eval, predictions)),
            }
        )

    return metrics


def evaluate_joint_holdout_discriminator(
    *,
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    text_columns: list[str],
    structured_weight: float = 1.0,
    text_weight_default: float = 0.01,
    text_weight_by_column: dict[str, float] | None = None,
    text_embedding_dim: int = 768,
    text_max_features: int = 50_000,
    random_state: int = 42,
) -> dict[str, Any]:
    text_weight_by_column = text_weight_by_column or {}

    feature_columns = list(real_train_df.columns)
    available_text_columns = [col for col in text_columns if col in feature_columns]
    structured_columns = [col for col in feature_columns if col not in set(available_text_columns)]

    real_x, holdout_x_real, real_diagnostics = _build_weighted_feature_matrices(
        reference_df=real_train_df,
        train_df=real_train_df[feature_columns],
        test_df=real_holdout_df[feature_columns],
        structured_columns=structured_columns,
        text_columns=available_text_columns,
        structured_weight=structured_weight,
        text_weight_default=text_weight_default,
        text_weight_by_column=text_weight_by_column,
        text_embedding_dim=text_embedding_dim,
        text_max_features=text_max_features,
    )
    synth_x, holdout_x_synth, synth_diagnostics = _build_weighted_feature_matrices(
        reference_df=real_train_df,
        train_df=synthetic_df[feature_columns],
        test_df=real_holdout_df[feature_columns],
        structured_columns=structured_columns,
        text_columns=available_text_columns,
        structured_weight=structured_weight,
        text_weight_default=text_weight_default,
        text_weight_by_column=text_weight_by_column,
        text_embedding_dim=text_embedding_dim,
        text_max_features=text_max_features,
    )

    real_metrics = _holdout_discriminator_metrics(real_x, holdout_x_real, random_state=random_state)
    synth_metrics = _holdout_discriminator_metrics(synth_x, holdout_x_synth, random_state=random_state)

    return {
        "embedding_backend": "tstr_weighted_structured_plus_tfidf_svd",
        "n_real_train": int(len(real_train_df)),
        "n_real_holdout": int(len(real_holdout_df)),
        "n_synthetic": int(len(synthetic_df)),
        "structured_columns": structured_columns,
        "text_columns": available_text_columns,
        "structured_block": real_diagnostics["structured_block"],
        "text_blocks": synth_diagnostics["text_blocks"],
        "total_feature_width": int(synth_x.shape[1]),
        "auc_real_vs_holdout": real_metrics["roc_auc"],
        "accuracy_real_vs_holdout": real_metrics["accuracy"],
        "auc_synth_vs_holdout": synth_metrics["roc_auc"],
        "accuracy_synth_vs_holdout": synth_metrics["accuracy"],
        "auc_drop_real_to_synth": float(synth_metrics["roc_auc"] - real_metrics["roc_auc"]),
        "accuracy_drop_real_to_synth": float(synth_metrics["accuracy"] - real_metrics["accuracy"]),
    }
