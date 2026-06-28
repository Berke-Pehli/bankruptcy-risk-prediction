"""Quantify validation-metric uncertainty with a paired company bootstrap.

The validation panel contains repeated annual observations for many companies. Sampling
firm-years independently would incorrectly treat those observations as unrelated. This
module instead samples company identifiers with replacement and retains every validation
year belonging to each sampled company.

Inputs:
    - Combined validation predictions for the six substantive models.
    - F2-optimal thresholds selected in the validation-evaluation stage.

Outputs:
    - Point estimates and 1,000 company-clustered bootstrap replicates.
    - Percentile confidence intervals for PR-AUC, ROC-AUC, Brier score, precision,
      recall, and F2.
    - Paired Random Forest performance differences against every competing model.
    - Figures for metric intervals and paired differences.

The same sampled companies are used for every model in a replicate, making pairwise
differences paired rather than independent. Selected classification thresholds remain
fixed throughout resampling, so precision, recall, and F2 intervals are conditional on
the validation-selected thresholds and do not include threshold-selection uncertainty.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from bankruptcy_risk.config import RANDOM_SEED
from bankruptcy_risk.evaluation import MODEL_LABELS, SUBSTANTIVE_MODELS
from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN

BOOTSTRAP_REPLICATES = 1_000
REFERENCE_MODEL = "random_forest"
METRICS = ("pr_auc", "roc_auc", "brier_score", "precision", "recall", "f2_score")
METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "brier_score": "Brier score",
    "precision": "Precision",
    "recall": "Recall",
    "f2_score": "F2 score",
}
HIGHER_IS_BETTER = {
    "pr_auc": True,
    "roc_auc": True,
    "brier_score": False,
    "precision": True,
    "recall": True,
    "f2_score": True,
}


def _prepare_prediction_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per firm-year with one probability column per model."""
    substantive = predictions.loc[
        predictions["model"].isin(SUBSTANTIVE_MODELS),
        [
            COMPANY_COLUMN,
            YEAR_COLUMN,
            TARGET_COLUMN,
            "model",
            "predicted_probability",
        ],
    ]
    matrix = (
        substantive.pivot(
            index=[COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN],
            columns="model",
            values="predicted_probability",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    missing_models = set(SUBSTANTIVE_MODELS).difference(matrix.columns)
    if missing_models or matrix.loc[:, SUBSTANTIVE_MODELS].isna().any().any():
        raise ValueError("Every firm-year must contain all substantive-model predictions.")
    if matrix.duplicated([COMPANY_COLUMN, YEAR_COLUMN]).any():
        raise ValueError("Bootstrap input must contain unique company-year keys.")
    return matrix


def _validate_thresholds(selected_thresholds: pd.DataFrame) -> dict[str, float]:
    """Return one valid fixed threshold for every substantive model."""
    required_columns = {"model", "selected_threshold"}
    missing_columns = required_columns.difference(selected_thresholds.columns)
    if missing_columns:
        raise ValueError(f"Selected-threshold columns are missing: {missing_columns}")
    if selected_thresholds["model"].duplicated().any():
        raise ValueError("Each model must have exactly one selected threshold.")
    threshold_map = selected_thresholds.set_index("model")["selected_threshold"].to_dict()
    if set(threshold_map) != set(SUBSTANTIVE_MODELS):
        raise ValueError("Thresholds must cover exactly the substantive models.")
    if any(not 0 <= threshold <= 1 for threshold in threshold_map.values()):
        raise ValueError("Selected thresholds must lie between zero and one.")
    return {model: float(threshold_map[model]) for model in SUBSTANTIVE_MODELS}


def draw_company_cluster_indices(
    company_ids: pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample companies with replacement and return all associated row positions."""
    if company_ids.empty:
        raise ValueError("Company-cluster sampling requires at least one observation.")
    groups = list(company_ids.groupby(company_ids, sort=False).indices.values())
    sampled_groups = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[group_index] for group_index in sampled_groups])


def _calculate_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate ranking, probability, and fixed-threshold classification metrics."""
    predicted = probabilities >= threshold
    positives = target == 1
    true_positive = int(np.sum(predicted & positives))
    false_positive = int(np.sum(predicted & ~positives))
    false_negative = int(np.sum(~predicted & positives))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f2_denominator = 4 * precision + recall
    f2_score = 5 * precision * recall / f2_denominator if f2_denominator else 0.0
    return {
        "pr_auc": average_precision_score(target, probabilities),
        "roc_auc": roc_auc_score(target, probabilities),
        "brier_score": brier_score_loss(target, probabilities),
        "precision": precision,
        "recall": recall,
        "f2_score": f2_score,
    }


def calculate_point_metrics(
    predictions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate original-sample metrics at fixed validation-selected thresholds."""
    matrix = _prepare_prediction_matrix(predictions)
    threshold_map = _validate_thresholds(selected_thresholds)
    target = matrix[TARGET_COLUMN].to_numpy()
    rows = []
    for model_name in SUBSTANTIVE_MODELS:
        values = _calculate_metrics(
            target,
            matrix[model_name].to_numpy(),
            threshold_map[model_name],
        )
        for metric, estimate in values.items():
            rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "point_estimate": estimate,
                    "threshold": threshold_map[model_name],
                    "threshold_conditional": metric in {"precision", "recall", "f2_score"},
                }
            )
    return pd.DataFrame(rows)


def generate_clustered_bootstrap_replicates(
    predictions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate paired company-clustered bootstrap metric replicates."""
    if n_bootstrap < 1:
        raise ValueError("At least one bootstrap replicate is required.")
    matrix = _prepare_prediction_matrix(predictions)
    threshold_map = _validate_thresholds(selected_thresholds)
    target = matrix[TARGET_COLUMN].to_numpy()
    probability_matrix = matrix.loc[:, SUBSTANTIVE_MODELS].to_numpy()
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, float | int | str]] = []
    completed = 0
    attempts = 0
    maximum_attempts = n_bootstrap * 10

    while completed < n_bootstrap and attempts < maximum_attempts:
        attempts += 1
        sampled_indices = draw_company_cluster_indices(matrix[COMPANY_COLUMN], rng)
        sampled_target = target[sampled_indices]
        if np.unique(sampled_target).size < 2:
            continue
        for model_index, model_name in enumerate(SUBSTANTIVE_MODELS):
            values = _calculate_metrics(
                sampled_target,
                probability_matrix[sampled_indices, model_index],
                threshold_map[model_name],
            )
            rows.append(
                {
                    "bootstrap_id": completed + 1,
                    "model": model_name,
                    **values,
                }
            )
        completed += 1

    if completed != n_bootstrap:
        raise RuntimeError("Unable to generate the requested valid bootstrap replicates.")
    return pd.DataFrame(rows)


def build_bootstrap_intervals(
    point_metrics: pd.DataFrame,
    bootstrap_replicates: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Build percentile confidence intervals for every model and metric."""
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must lie strictly between zero and one.")
    tail_probability = (1 - confidence_level) / 2
    rows = []
    for point in point_metrics.itertuples(index=False):
        values = bootstrap_replicates.loc[
            bootstrap_replicates["model"].eq(point.model),
            point.metric,
        ]
        rows.append(
            {
                **point._asdict(),
                "bootstrap_mean": values.mean(),
                "bootstrap_standard_error": values.std(ddof=1),
                "confidence_interval_lower": values.quantile(tail_probability),
                "confidence_interval_upper": values.quantile(1 - tail_probability),
                "confidence_level": confidence_level,
                "bootstrap_replicates": values.size,
                "resampling_unit": "company",
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_differences(
    point_metrics: pd.DataFrame,
    bootstrap_replicates: pd.DataFrame,
    reference_model: str = REFERENCE_MODEL,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compare a reference model with competitors using paired bootstrap draws."""
    if reference_model not in SUBSTANTIVE_MODELS:
        raise ValueError("Reference model must be one of the substantive models.")
    tail_probability = (1 - confidence_level) / 2
    rows = []
    point_lookup = point_metrics.set_index(["model", "metric"])["point_estimate"]
    competitors = [model for model in SUBSTANTIVE_MODELS if model != reference_model]

    for metric in METRICS:
        replicate_matrix = bootstrap_replicates.pivot(
            index="bootstrap_id",
            columns="model",
            values=metric,
        )
        for competitor in competitors:
            differences = replicate_matrix[reference_model] - replicate_matrix[competitor]
            point_difference = point_lookup[reference_model, metric] - point_lookup[
                competitor, metric
            ]
            lower = differences.quantile(tail_probability)
            upper = differences.quantile(1 - tail_probability)
            reference_better = (
                differences.gt(0) if HIGHER_IS_BETTER[metric] else differences.lt(0)
            )
            rows.append(
                {
                    "reference_model": reference_model,
                    "reference_label": MODEL_LABELS[reference_model],
                    "competitor_model": competitor,
                    "competitor_label": MODEL_LABELS[competitor],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    "point_difference_reference_minus_competitor": point_difference,
                    "bootstrap_mean_difference": differences.mean(),
                    "confidence_interval_lower": lower,
                    "confidence_interval_upper": upper,
                    "confidence_level": confidence_level,
                    "confidence_interval_excludes_zero": bool(lower > 0 or upper < 0),
                    "probability_reference_better": reference_better.mean(),
                    "bootstrap_replicates": differences.size,
                    "resampling_unit": "company",
                }
            )
    return pd.DataFrame(rows)


def plot_bootstrap_metric_intervals(intervals: pd.DataFrame) -> Figure:
    """Plot confidence intervals for PR-AUC, ROC-AUC, and fixed-threshold F2."""
    sns.set_theme(style="whitegrid", context="notebook")
    displayed_metrics = ("pr_auc", "roc_auc", "f2_score")
    figure, axes = plt.subplots(1, 3, figsize=(17, 7))
    for axis, metric in zip(axes, displayed_metrics, strict=True):
        displayed = intervals.loc[intervals["metric"].eq(metric)].copy()
        displayed["model_order"] = displayed["model"].map(
            {model: index for index, model in enumerate(SUBSTANTIVE_MODELS)}
        )
        displayed = displayed.sort_values("model_order", ascending=False)
        lower_error = displayed["point_estimate"] - displayed["confidence_interval_lower"]
        upper_error = displayed["confidence_interval_upper"] - displayed["point_estimate"]
        axis.errorbar(
            displayed["point_estimate"],
            displayed["model_label"],
            xerr=np.vstack([lower_error, upper_error]),
            fmt="o",
            color="#1F4E79",
            ecolor="#666666",
            capsize=4,
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Point estimate and 95% interval")
        axis.set_xlim(0, 1)
    figure.suptitle("Company-Clustered Bootstrap Validation Uncertainty", y=1.02)
    figure.tight_layout()
    return figure


def plot_pairwise_differences(pairwise: pd.DataFrame) -> Figure:
    """Plot paired Random Forest differences for key validation metrics."""
    sns.set_theme(style="whitegrid", context="notebook")
    displayed_metrics = ("pr_auc", "roc_auc", "f2_score")
    figure, axes = plt.subplots(1, 3, figsize=(17, 7))
    for axis, metric in zip(axes, displayed_metrics, strict=True):
        displayed = pairwise.loc[pairwise["metric"].eq(metric)].copy()
        lower_error = (
            displayed["point_difference_reference_minus_competitor"]
            - displayed["confidence_interval_lower"]
        )
        upper_error = (
            displayed["confidence_interval_upper"]
            - displayed["point_difference_reference_minus_competitor"]
        )
        axis.errorbar(
            displayed["point_difference_reference_minus_competitor"],
            displayed["competitor_label"],
            xerr=np.vstack([lower_error, upper_error]),
            fmt="o",
            color="#C55A11",
            ecolor="#666666",
            capsize=4,
        )
        axis.axvline(0, color="#333333", linestyle="--")
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Random Forest minus competitor")
    figure.suptitle("Paired Company-Bootstrap Performance Differences", y=1.02)
    figure.tight_layout()
    return figure
