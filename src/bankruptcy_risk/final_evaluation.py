"""Freeze the champion workflow and evaluate it once on the final test period.

This module marks the transition from model development to final assessment. Random
Forest is selected using validation PR-AUC, while its hyperparameters and operational
threshold come from earlier training and validation tasks. Those choices are frozen
before any 2015-2018 test outcome contributes to a metric.

Inputs:
    - Validation model metrics, selected Random Forest hyperparameters, and F2 threshold.
    - The temporally labelled model dataset.

Process:
    - Refit the frozen 500-tree pipeline on all 1999-2014 development observations.
    - Score the untouched 2015-2018 test features once.
    - Evaluate ranking, probability quality, and fixed-threshold classification results.
    - Quantify uncertainty with 1,000 company-clustered test bootstrap samples.

Outputs:
    - Frozen specification and fitted final model.
    - Test predictions, metrics, calibration, bootstrap intervals, and figures.
    - A validation-versus-test comparison for transparent generalization assessment.

The frozen F2 threshold is not re-optimized after development refitting. Test outcomes
are used only for the final metrics and uncertainty estimates reported by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from bankruptcy_risk.evaluation import SUBSTANTIVE_MODELS, evaluate_predictions
from bankruptcy_risk.features import MODEL_FEATURES
from bankruptcy_risk.random_forest import FINAL_ESTIMATORS, make_random_forest_pipeline
from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN
from bankruptcy_risk.uncertainty import draw_company_cluster_indices

FINAL_MODEL_NAME = "random_forest"
FINAL_BOOTSTRAP_REPLICATES = 1_000
FINAL_METRICS = ("pr_auc", "roc_auc", "brier_score", "precision", "recall", "f2_score")


@dataclass(frozen=True)
class FinalModelSpecification:
    """Store every model and decision choice frozen before final testing."""

    model: str
    champion_selection_metric: str
    validation_pr_auc: float
    max_depth: int
    min_samples_leaf: int
    max_features: float
    n_estimators: int
    selected_threshold: float
    threshold_selection_rule: str


def freeze_final_specification(
    validation_metrics: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    random_forest_selection: pd.DataFrame,
    n_estimators: int = FINAL_ESTIMATORS,
) -> FinalModelSpecification:
    """Freeze the champion, hyperparameters, and threshold from development evidence."""
    substantive = validation_metrics.loc[
        validation_metrics["model"].isin(SUBSTANTIVE_MODELS)
    ]
    if set(substantive["model"]) != set(SUBSTANTIVE_MODELS):
        raise ValueError("Validation metrics must cover every substantive model.")
    champion_row = substantive.sort_values(
        ["pr_auc", "model"],
        ascending=[False, True],
    ).iloc[0]
    if champion_row["model"] != FINAL_MODEL_NAME:
        raise ValueError("Random Forest is not the validation PR-AUC champion.")

    threshold_row = selected_thresholds.loc[
        selected_thresholds["model"].eq(FINAL_MODEL_NAME)
    ]
    if len(threshold_row) != 1 or len(random_forest_selection) != 1:
        raise ValueError("Exactly one frozen threshold and Random Forest selection are required.")
    threshold_row = threshold_row.iloc[0]
    selection_row = random_forest_selection.iloc[0]
    threshold = float(threshold_row["selected_threshold"])
    if not 0 <= threshold <= 1 or n_estimators < 1:
        raise ValueError("The frozen threshold and estimator count must be valid.")

    return FinalModelSpecification(
        model=FINAL_MODEL_NAME,
        champion_selection_metric="validation_pr_auc",
        validation_pr_auc=float(champion_row["pr_auc"]),
        max_depth=int(selection_row["max_depth"]),
        min_samples_leaf=int(selection_row["min_samples_leaf"]),
        max_features=float(selection_row["max_features"]),
        n_estimators=n_estimators,
        selected_threshold=threshold,
        threshold_selection_rule=str(threshold_row["selection_rule"]),
    )


def fit_final_champion(
    model_data: pd.DataFrame,
    specification: FinalModelSpecification,
) -> Pipeline:
    """Refit the frozen pipeline on the 1999-2014 development observations."""
    development = model_data.loc[model_data["sample_period"].isin(["train", "validation"])]
    if development.empty or model_data.loc[model_data["sample_period"].eq("test")].empty:
        raise ValueError("Both development and test observations are required.")
    pipeline = make_random_forest_pipeline(
        max_depth=specification.max_depth,
        min_samples_leaf=specification.min_samples_leaf,
        max_features=specification.max_features,
        n_estimators=specification.n_estimators,
    )
    pipeline.fit(development.loc[:, MODEL_FEATURES], development[TARGET_COLUMN])
    return pipeline


def create_final_test_predictions(
    model_data: pd.DataFrame,
    fitted_model: Pipeline,
    specification: FinalModelSpecification,
) -> pd.DataFrame:
    """Score only the reserved 2015-2018 test observations."""
    test = model_data.loc[model_data["sample_period"].eq("test")]
    if test.empty:
        raise ValueError("Final test observations are required.")
    probabilities = fitted_model.predict_proba(test.loc[:, MODEL_FEATURES])[:, 1]
    predictions = test.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN]].copy()
    predictions["model"] = specification.model
    predictions["predicted_probability"] = probabilities
    predictions["frozen_threshold"] = specification.selected_threshold
    predictions["predicted_class"] = (
        probabilities >= specification.selected_threshold
    ).astype("int8")
    return predictions.reset_index(drop=True)


def evaluate_final_test_predictions(
    predictions: pd.DataFrame,
    specification: FinalModelSpecification,
) -> pd.DataFrame:
    """Report test metrics at default and frozen validation-selected thresholds."""
    default_metrics = evaluate_predictions(
        predictions,
        thresholds=0.5,
        models=(FINAL_MODEL_NAME,),
        threshold_source="default_0.50_reference",
    )
    frozen_metrics = evaluate_predictions(
        predictions,
        thresholds={FINAL_MODEL_NAME: specification.selected_threshold},
        models=(FINAL_MODEL_NAME,),
        threshold_source="frozen_validation_f2",
    )
    metrics = pd.concat([default_metrics, frozen_metrics], ignore_index=True)
    metrics.insert(0, "evaluation_period", "final_test_2015_2018")
    metrics["model_refit_period"] = "development_1999_2014"
    metrics["test_used_for_selection"] = False
    return metrics


def build_final_model_audit(
    model_data: pd.DataFrame,
    specification: FinalModelSpecification,
) -> pd.DataFrame:
    """Document frozen choices and chronological sample boundaries."""
    development = model_data.loc[model_data["sample_period"].isin(["train", "validation"])]
    test = model_data.loc[model_data["sample_period"].eq("test")]
    return pd.DataFrame(
        [
            {
                **asdict(specification),
                "development_start_year": int(development[YEAR_COLUMN].min()),
                "development_end_year": int(development[YEAR_COLUMN].max()),
                "development_observations": len(development),
                "development_bankruptcies": int(development[TARGET_COLUMN].sum()),
                "test_start_year": int(test[YEAR_COLUMN].min()),
                "test_end_year": int(test[YEAR_COLUMN].max()),
                "test_observations": len(test),
                "test_bankruptcies": int(test[TARGET_COLUMN].sum()),
                "test_used_for_model_selection": False,
                "test_used_for_hyperparameter_selection": False,
                "test_used_for_threshold_selection": False,
            }
        ]
    )


def build_final_test_calibration(
    predictions: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize final-test calibration in predicted-probability quantile bins."""
    if n_bins < 2:
        raise ValueError("Calibration requires at least two bins.")
    data = predictions.loc[:, [TARGET_COLUMN, "predicted_probability"]].copy()
    data["calibration_bin"] = pd.qcut(
        data["predicted_probability"],
        q=min(n_bins, data["predicted_probability"].nunique()),
        labels=False,
        duplicates="drop",
    )
    calibration = (
        data.groupby("calibration_bin", observed=True)
        .agg(
            observations=(TARGET_COLUMN, "size"),
            bankruptcies=(TARGET_COLUMN, "sum"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_event_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )
    calibration["calibration_bin"] = calibration["calibration_bin"].astype(int) + 1
    return calibration


def _calculate_final_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate metrics used by the final company bootstrap."""
    predicted = probabilities >= threshold
    positives = target == 1
    true_positive = int(np.sum(predicted & positives))
    false_positive = int(np.sum(predicted & ~positives))
    false_negative = int(np.sum(~predicted & positives))
    precision = true_positive / (true_positive + false_positive) if predicted.sum() else 0.0
    recall = true_positive / (true_positive + false_negative) if positives.sum() else 0.0
    denominator = 4 * precision + recall
    return {
        "pr_auc": average_precision_score(target, probabilities),
        "roc_auc": roc_auc_score(target, probabilities),
        "brier_score": brier_score_loss(target, probabilities),
        "precision": precision,
        "recall": recall,
        "f2_score": 5 * precision * recall / denominator if denominator else 0.0,
    }


def generate_final_test_bootstrap(
    predictions: pd.DataFrame,
    specification: FinalModelSpecification,
    n_bootstrap: int = FINAL_BOOTSTRAP_REPLICATES,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate company-clustered final-test metric replicates."""
    if n_bootstrap < 1:
        raise ValueError("At least one final-test bootstrap replicate is required.")
    rng = np.random.default_rng(random_seed)
    target = predictions[TARGET_COLUMN].to_numpy()
    probabilities = predictions["predicted_probability"].to_numpy()
    rows = []
    completed = 0
    attempts = 0
    while completed < n_bootstrap and attempts < n_bootstrap * 10:
        attempts += 1
        sampled = draw_company_cluster_indices(predictions[COMPANY_COLUMN], rng)
        sampled_target = target[sampled]
        if np.unique(sampled_target).size < 2:
            continue
        rows.append(
            {
                "bootstrap_id": completed + 1,
                **_calculate_final_metrics(
                    sampled_target,
                    probabilities[sampled],
                    specification.selected_threshold,
                ),
            }
        )
        completed += 1
    if completed != n_bootstrap:
        raise RuntimeError("Unable to generate valid final-test bootstrap replicates.")
    return pd.DataFrame(rows)


def build_final_test_intervals(
    final_metrics: pd.DataFrame,
    bootstrap_replicates: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Build percentile intervals around frozen-threshold final-test metrics."""
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must lie strictly between zero and one.")
    frozen = final_metrics.loc[final_metrics["threshold_source"].eq("frozen_validation_f2")]
    if len(frozen) != 1:
        raise ValueError("Exactly one frozen-threshold metric row is required.")
    frozen = frozen.iloc[0]
    tail = (1 - confidence_level) / 2
    rows = []
    for metric in FINAL_METRICS:
        values = bootstrap_replicates[metric]
        rows.append(
            {
                "model": FINAL_MODEL_NAME,
                "metric": metric,
                "point_estimate": frozen[metric],
                "bootstrap_mean": values.mean(),
                "bootstrap_standard_error": values.std(ddof=1),
                "confidence_interval_lower": values.quantile(tail),
                "confidence_interval_upper": values.quantile(1 - tail),
                "confidence_level": confidence_level,
                "bootstrap_replicates": len(values),
                "resampling_unit": "company",
                "threshold": frozen["threshold"],
                "threshold_source": "frozen_validation_f2",
            }
        )
    return pd.DataFrame(rows)


def build_validation_test_comparison(
    validation_optimized_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Place champion validation and final-test results in one auditable table."""
    validation = validation_optimized_metrics.loc[
        validation_optimized_metrics["model"].eq(FINAL_MODEL_NAME)
    ].copy()
    test = final_metrics.loc[
        final_metrics["threshold_source"].eq("frozen_validation_f2")
    ].copy()
    if len(validation) != 1 or len(test) != 1:
        raise ValueError("One validation and one final-test champion row are required.")
    columns = [
        "model",
        "threshold",
        "observations",
        "bankruptcies",
        "predicted_bankruptcies",
        "precision",
        "recall",
        "f1_score",
        "f2_score",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
    ]
    validation = validation.loc[:, columns].assign(evaluation_period="validation_2012_2014")
    test = test.loc[:, columns].assign(evaluation_period="final_test_2015_2018")
    return pd.concat([validation, test], ignore_index=True).loc[
        :, ["evaluation_period", *columns]
    ]


def plot_final_test_diagnostics(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
) -> Figure:
    """Plot final-test precision-recall, ROC, and calibration diagnostics."""
    sns.set_theme(style="whitegrid", context="notebook")
    target = predictions[TARGET_COLUMN]
    probabilities = predictions["predicted_probability"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 6))

    precision, recall, _ = precision_recall_curve(target, probabilities)
    axes[0].plot(recall, precision, color="#1F4E79", linewidth=2)
    axes[0].axhline(target.mean(), color="#666666", linestyle="--")
    axes[0].set_title(f"Precision-Recall (AP={average_precision_score(target, probabilities):.3f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    false_positive_rate, true_positive_rate, _ = roc_curve(target, probabilities)
    axes[1].plot(false_positive_rate, true_positive_rate, color="#1F4E79", linewidth=2)
    axes[1].plot([0, 1], [0, 1], color="#666666", linestyle="--")
    axes[1].set_title(f"ROC (AUC={roc_auc_score(target, probabilities):.3f})")
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    axes[2].plot(
        calibration["mean_predicted_probability"],
        calibration["observed_event_rate"],
        color="#C55A11",
        marker="o",
        linewidth=2,
    )
    maximum = max(
        calibration["mean_predicted_probability"].max(),
        calibration["observed_event_rate"].max(),
    )
    axes[2].plot([0, maximum], [0, maximum], color="#666666", linestyle="--")
    axes[2].set_title("Probability Calibration")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed event rate")

    figure.suptitle("Final Test Evaluation: 2015-2018", y=1.03)
    figure.tight_layout()
    return figure


def plot_final_test_confusion_matrix(
    predictions: pd.DataFrame,
    specification: FinalModelSpecification,
) -> Figure:
    """Plot the final confusion matrix at the frozen validation threshold."""
    target = predictions[TARGET_COLUMN]
    predicted = predictions["predicted_probability"].ge(specification.selected_threshold)
    matrix = confusion_matrix(target, predicted, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No bankruptcy", "Bankruptcy"],
        yticklabels=["No bankruptcy", "Bankruptcy"],
        ax=axis,
    )
    axis.set_title(
        f"Final Test Confusion Matrix\nFrozen threshold={specification.selected_threshold:.4f}"
    )
    axis.set_xlabel("Predicted outcome")
    axis.set_ylabel("Observed outcome")
    figure.tight_layout()
    return figure
