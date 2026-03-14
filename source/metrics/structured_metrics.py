from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


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

    _, metrics = qa.report(
        syn_tgt_data=synthetic_eval,
        trn_tgt_data=train_eval,
        hol_tgt_data=holdout_eval,
    )

    report_file = Path("model-report.html")
    if report_file.exists():
        report_file.unlink()

    return asdict(_to_summary(metrics))
