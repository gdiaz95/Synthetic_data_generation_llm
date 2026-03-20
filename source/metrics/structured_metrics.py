from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from source.metrics.free_text_metrics import _embed_tfidf_svd_many


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


def _infer_prediction_task(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        raise ValueError("Target column has no non-null values; cannot run TSTR.")

    if pd.api.types.is_bool_dtype(non_null):
        return "classification"

    if pd.api.types.is_numeric_dtype(non_null):
        unique_count = int(non_null.nunique(dropna=True))
        if unique_count <= 20:
            return "classification"
        return "regression"

    return "classification"


def _clean_text_series_for_tstr(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _normalize_block(block: np.ndarray) -> np.ndarray:
    if block.ndim != 2 or block.shape[1] == 0:
        return block
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return block / norms


def _build_structured_block(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            )
        )

    if not transformers:
        empty_train = np.empty((len(train_df), 0), dtype=np.float64)
        empty_test = np.empty((len(test_df), 0), dtype=np.float64)
        return empty_train, empty_test, 0

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    train_block = preprocessor.fit_transform(train_df)
    test_block = preprocessor.transform(test_df)

    train_block = np.asarray(train_block, dtype=np.float64)
    test_block = np.asarray(test_block, dtype=np.float64)
    return train_block, test_block, int(train_block.shape[1])


def evaluate_tstr(
    *,
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_column: str,
    text_columns: list[str],
    structured_weight: float = 1.0,
    text_weight_default: float = 0.3,
    text_weight_by_column: dict[str, float] | None = None,
    text_embedding_dim: int = 768,
    text_max_features: int = 5_000,
) -> dict[str, Any]:
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' missing in real train dataframe.")
    if target_column not in synthetic_df.columns:
        raise ValueError(f"Target column '{target_column}' missing in synthetic dataframe.")
    if target_column not in holdout_df.columns:
        raise ValueError(f"Target column '{target_column}' missing in real holdout dataframe.")

    feature_columns = [c for c in train_df.columns if c != target_column]
    if not feature_columns:
        raise ValueError("No feature columns available for TSTR after excluding target column.")

    text_features = [c for c in text_columns if c in feature_columns]
    numeric_features = [
        c for c in feature_columns
        if c not in text_features and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    categorical_features = [
        c for c in feature_columns if c not in text_features and c not in numeric_features
    ]

    task_type = _infer_prediction_task(train_df[target_column])
    x_train = synthetic_df[feature_columns].copy()
    y_train = synthetic_df[target_column]
    x_test = holdout_df[feature_columns].copy()
    y_test = holdout_df[target_column]

    train_mask = y_train.notna()
    test_mask = y_test.notna()
    x_train = x_train.loc[train_mask]
    y_train = y_train.loc[train_mask]
    x_test = x_test.loc[test_mask]
    y_test = y_test.loc[test_mask]

    if x_train.empty or x_test.empty:
        raise ValueError("Insufficient non-null target rows for TSTR.")

    structured_train, structured_test, structured_width = _build_structured_block(
        train_df=x_train,
        test_df=x_test,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    if structured_width == 0 and not text_features:
        raise ValueError("No valid transformers could be built for TSTR features.")

    text_weight_by_column = text_weight_by_column or {}
    feature_blocks_train: list[np.ndarray] = []
    feature_blocks_test: list[np.ndarray] = []
    text_block_details: dict[str, Any] = {}

    if structured_width > 0:
        structured_train = _normalize_block(structured_train)
        structured_test = _normalize_block(structured_test)
        feature_blocks_train.append(structured_train * float(structured_weight))
        feature_blocks_test.append(structured_test * float(structured_weight))

    reference_width = max(1, structured_width if structured_width > 0 else len(feature_columns) - len(text_features))

    for text_col in text_features:
        embedded_sets, latent_dim = _embed_tfidf_svd_many(
            [
                _clean_text_series_for_tstr(x_train[text_col]),
                _clean_text_series_for_tstr(x_test[text_col]),
            ],
            embedding_dim=text_embedding_dim,
            max_features=text_max_features,
            min_df=1,
        )
        text_train, text_test = embedded_sets
        text_train = _normalize_block(text_train)
        text_test = _normalize_block(text_test)

        text_width = int(text_train.shape[1])
        column_weight = float(text_weight_by_column.get(text_col, text_weight_default))
        normalization_factor = 1.0
        if text_width > 0:
            normalization_factor = float(np.sqrt(reference_width / text_width))
        effective_weight = column_weight * normalization_factor

        feature_blocks_train.append(text_train * effective_weight)
        feature_blocks_test.append(text_test * effective_weight)
        text_block_details[text_col] = {
            "raw_width": text_width,
            "normalization_factor": normalization_factor,
            "configured_weight": column_weight,
            "effective_weight": effective_weight,
            "svd_components": int(latent_dim),
        }

    train_matrix = np.hstack(feature_blocks_train) if feature_blocks_train else np.empty((len(x_train), 0))
    test_matrix = np.hstack(feature_blocks_test) if feature_blocks_test else np.empty((len(x_test), 0))
    if train_matrix.shape[1] == 0:
        raise ValueError("TSTR produced an empty feature matrix.")

    if task_type == "classification":
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
    else:
        model = Ridge(alpha=1.0)

    model.fit(train_matrix, y_train)
    preds = model.predict(test_matrix)

    if task_type == "classification":
        result = {
            "task_type": task_type,
            "target_column": target_column,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "model_type": "linear",
            "accuracy": float(accuracy_score(y_test, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "f1_weighted": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
            "structured_block": {
                "width": structured_width,
                "configured_weight": float(structured_weight),
            },
            "text_blocks": text_block_details,
            "feature_matrix_width": int(train_matrix.shape[1]),
        }
        if len(pd.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(test_matrix)[:, 1]
            result["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        return result

    return {
        "task_type": task_type,
        "target_column": target_column,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "model_type": "linear",
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "structured_block": {
            "width": structured_width,
            "configured_weight": float(structured_weight),
        },
        "text_blocks": text_block_details,
        "feature_matrix_width": int(train_matrix.shape[1]),
    }
