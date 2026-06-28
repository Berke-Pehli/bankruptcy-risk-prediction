"""Compare validation predictions and select operational decision thresholds.

This module brings every benchmark into one consistent evaluation framework. It checks
that models predict the same validation firm-years and then measures discrimination,
probability accuracy, and classification performance.

Inputs:
    - Validation probabilities from the baseline, Logistic Regression, pruned-tree,
      Random Forest, and Gradient Boosting tasks.

Outputs:
    - A combined and audited validation-prediction table.
    - Metrics at the conventional 0.50 threshold for all eight models.
    - Precision-recall threshold curves for the six substantive models.
    - One F2-optimal threshold per substantive model and its associated metrics.
    - Calibration-bin data and four model-comparison figures.

F2 gives recall twice the weight of precision because failing to flag a future
bankruptcy is assumed to be more costly than investigating a false alarm. Thresholds
are selected on 2012-2014 validation observations and must therefore be treated as model
selection choices. They are not final unbiased performance estimates; the 2015-2018
test period remains untouched.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN

MODEL_ORDER = (
    "majority_non_bankrupt",
    "training_prevalence",
    "interpretable_logit",
    "ridge_logit",
    "lasso_logit",
    "pruned_decision_tree",
    "random_forest",
    "gradient_boosting",
)
SUBSTANTIVE_MODELS = (
    "interpretable_logit",
    "ridge_logit",
    "lasso_logit",
    "pruned_decision_tree",
    "random_forest",
    "gradient_boosting",
)
CALIBRATION_MODELS = ("training_prevalence", *SUBSTANTIVE_MODELS)
MODEL_LABELS = {
    "majority_non_bankrupt": "Majority baseline",
    "training_prevalence": "Prevalence baseline",
    "interpretable_logit": "Interpretable Logit",
    "ridge_logit": "Ridge Logit",
    "lasso_logit": "Lasso Logit",
    "pruned_decision_tree": "Pruned Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}
PREDICTION_COLUMNS = (
    COMPANY_COLUMN,
    YEAR_COLUMN,
    TARGET_COLUMN,
    "model",
    "predicted_probability",
)


def combine_validation_predictions(
    prediction_tables: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Combine model predictions after checking keys, outcomes, and probabilities.

    Every expected model must contain one prediction for exactly the same company-year
    keys. The observed target must also agree across all copies of each observation.
    """
    tables = [table.loc[:, PREDICTION_COLUMNS].copy() for table in prediction_tables]
    if not tables:
        raise ValueError("At least one validation-prediction table is required.")
    combined = pd.concat(tables, ignore_index=True)

    if set(combined["model"]) != set(MODEL_ORDER):
        raise ValueError("Validation predictions do not contain the expected model set.")
    if combined.duplicated(["model", COMPANY_COLUMN, YEAR_COLUMN]).any():
        raise ValueError("Each model must have one prediction per company-year.")
    if not np.isfinite(combined["predicted_probability"]).all():
        raise ValueError("Predicted probabilities must be finite.")
    if not combined["predicted_probability"].between(0, 1).all():
        raise ValueError("Predicted probabilities must lie between zero and one.")
    if not combined[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Observed outcomes must contain only zero and one.")

    key_columns = [COMPANY_COLUMN, YEAR_COLUMN]
    reference = combined.loc[
        combined["model"].eq(MODEL_ORDER[0]),
        [*key_columns, TARGET_COLUMN],
    ].sort_values(key_columns, ignore_index=True)
    for model_name in MODEL_ORDER[1:]:
        model_rows = combined.loc[
            combined["model"].eq(model_name),
            [*key_columns, TARGET_COLUMN],
        ].sort_values(key_columns, ignore_index=True)
        if not model_rows.equals(reference):
            raise ValueError(f"Validation keys or outcomes differ for {model_name}.")

    model_order = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
    combined = combined.assign(_model_order=model_order).sort_values(
        ["_model_order", YEAR_COLUMN, COMPANY_COLUMN],
        ignore_index=True,
    )
    return combined.drop(columns="_model_order")


def evaluate_predictions(
    predictions: pd.DataFrame,
    thresholds: float | Mapping[str, float] = 0.5,
    models: Sequence[str] = MODEL_ORDER,
    threshold_source: str = "default_0.50",
) -> pd.DataFrame:
    """Calculate discrimination, calibration, and classification metrics by model."""
    rows: list[dict[str, float | int | str]] = []
    for model_name in models:
        model_predictions = predictions.loc[predictions["model"].eq(model_name)]
        if model_predictions.empty:
            raise ValueError(f"Predictions are missing for {model_name}.")
        threshold = (
            float(thresholds[model_name])
            if isinstance(thresholds, Mapping)
            else float(thresholds)
        )
        if not 0 <= threshold <= 1:
            raise ValueError("Classification thresholds must lie between zero and one.")

        target = model_predictions[TARGET_COLUMN].to_numpy()
        probabilities = model_predictions["predicted_probability"].to_numpy()
        predicted_class = (probabilities >= threshold).astype(int)
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            target,
            predicted_class,
            labels=[0, 1],
        ).ravel()
        specificity = true_negative / (true_negative + false_positive)

        rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "threshold": threshold,
                "threshold_source": threshold_source,
                "observations": len(target),
                "bankruptcies": int(target.sum()),
                "predicted_bankruptcies": int(predicted_class.sum()),
                "true_negative": int(true_negative),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
                "accuracy": accuracy_score(target, predicted_class),
                "balanced_accuracy": balanced_accuracy_score(target, predicted_class),
                "precision": precision_score(target, predicted_class, zero_division=0),
                "recall": recall_score(target, predicted_class, zero_division=0),
                "specificity": specificity,
                "f1_score": f1_score(target, predicted_class, zero_division=0),
                "f2_score": fbeta_score(
                    target,
                    predicted_class,
                    beta=2,
                    zero_division=0,
                ),
                "roc_auc": roc_auc_score(target, probabilities),
                "pr_auc": average_precision_score(target, probabilities),
                "brier_score": brier_score_loss(target, probabilities),
                "log_loss": log_loss(target, probabilities, labels=[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_curves(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate precision, recall, F1, and F2 across substantive-model thresholds."""
    rows: list[pd.DataFrame] = []
    for model_name in SUBSTANTIVE_MODELS:
        model_predictions = predictions.loc[predictions["model"].eq(model_name)]
        target = model_predictions[TARGET_COLUMN].to_numpy()
        probabilities = model_predictions["predicted_probability"].to_numpy()
        precision, recall, thresholds = precision_recall_curve(target, probabilities)
        precision = precision[:-1]
        recall = recall[:-1]
        f1_denominator = precision + recall
        f2_denominator = 4 * precision + recall
        f1 = np.divide(
            2 * precision * recall,
            f1_denominator,
            out=np.zeros_like(precision),
            where=f1_denominator > 0,
        )
        f2 = np.divide(
            5 * precision * recall,
            f2_denominator,
            out=np.zeros_like(precision),
            where=f2_denominator > 0,
        )
        rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "threshold": thresholds,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "f2_score": f2,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def select_f2_thresholds(threshold_curves: pd.DataFrame) -> pd.DataFrame:
    """Select the highest threshold attaining each model's maximum validation F2."""
    selected_rows = []
    for model_name in SUBSTANTIVE_MODELS:
        model_curve = threshold_curves.loc[threshold_curves["model"].eq(model_name)]
        if model_curve.empty:
            raise ValueError(f"Threshold curve is missing for {model_name}.")
        selected = model_curve.sort_values(
            ["f2_score", "threshold"],
            ascending=[False, False],
        ).iloc[0]
        selected_rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "selected_threshold": selected["threshold"],
                "validation_precision": selected["precision"],
                "validation_recall": selected["recall"],
                "validation_f1_score": selected["f1_score"],
                "validation_f2_score": selected["f2_score"],
                "selection_rule": "maximum_validation_f2",
                "beta": 2,
            }
        )
    return pd.DataFrame(selected_rows)


def build_calibration_table(
    predictions: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize observed and predicted bankruptcy rates in quantile bins."""
    if n_bins < 2:
        raise ValueError("Calibration requires at least two requested bins.")
    rows: list[pd.DataFrame] = []
    for model_name in CALIBRATION_MODELS:
        model_predictions = predictions.loc[
            predictions["model"].eq(model_name),
            [TARGET_COLUMN, "predicted_probability"],
        ].copy()
        unique_probabilities = model_predictions["predicted_probability"].nunique()
        if unique_probabilities == 1:
            model_predictions["calibration_bin"] = 0
        else:
            requested_bins = min(n_bins, unique_probabilities)
            model_predictions["calibration_bin"] = pd.qcut(
                model_predictions["predicted_probability"],
                q=requested_bins,
                labels=False,
                duplicates="drop",
            )
        calibration = (
            model_predictions.groupby("calibration_bin", observed=True)
            .agg(
                observations=(TARGET_COLUMN, "size"),
                bankruptcies=(TARGET_COLUMN, "sum"),
                mean_predicted_probability=("predicted_probability", "mean"),
                observed_event_rate=(TARGET_COLUMN, "mean"),
            )
            .reset_index()
        )
        calibration["calibration_bin"] = calibration["calibration_bin"].astype(int) + 1
        calibration.insert(0, "model", model_name)
        rows.append(calibration)
    return pd.concat(rows, ignore_index=True)


def _apply_plot_style() -> None:
    """Apply the restrained visual style used by evaluation figures."""
    sns.set_theme(style="whitegrid", context="notebook")


def plot_precision_recall_curves(predictions: pd.DataFrame) -> Figure:
    """Plot validation precision-recall curves for substantive models."""
    _apply_plot_style()
    figure, axis = plt.subplots(figsize=(10, 7))
    event_rate = predictions.loc[
        predictions["model"].eq(SUBSTANTIVE_MODELS[0]), TARGET_COLUMN
    ].mean()
    for model_name in SUBSTANTIVE_MODELS:
        model_predictions = predictions.loc[predictions["model"].eq(model_name)]
        target = model_predictions[TARGET_COLUMN]
        probabilities = model_predictions["predicted_probability"]
        precision, recall, _ = precision_recall_curve(target, probabilities)
        average_precision = average_precision_score(target, probabilities)
        axis.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{MODEL_LABELS[model_name]} (AP={average_precision:.3f})",
        )
    axis.axhline(
        event_rate,
        color="#666666",
        linestyle="--",
        label=f"Validation prevalence ({event_rate:.3f})",
    )
    axis.set_title("Validation Precision-Recall Curves")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper right", fontsize=9)
    figure.tight_layout()
    return figure


def plot_roc_curves(predictions: pd.DataFrame) -> Figure:
    """Plot validation ROC curves for substantive models."""
    _apply_plot_style()
    figure, axis = plt.subplots(figsize=(10, 7))
    for model_name in SUBSTANTIVE_MODELS:
        model_predictions = predictions.loc[predictions["model"].eq(model_name)]
        target = model_predictions[TARGET_COLUMN]
        probabilities = model_predictions["predicted_probability"]
        false_positive_rate, true_positive_rate, _ = roc_curve(target, probabilities)
        auc = roc_auc_score(target, probabilities)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{MODEL_LABELS[model_name]} (AUC={auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", label="Random ranking")
    axis.set_title("Validation ROC Curves")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    return figure


def plot_calibration_curves(calibration: pd.DataFrame) -> Figure:
    """Plot observed against predicted validation bankruptcy rates."""
    _apply_plot_style()
    figure, axis = plt.subplots(figsize=(10, 7))
    for model_name in CALIBRATION_MODELS:
        model_calibration = calibration.loc[calibration["model"].eq(model_name)]
        axis.plot(
            model_calibration["mean_predicted_probability"],
            model_calibration["observed_event_rate"],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS[model_name],
        )
    maximum = max(
        calibration["mean_predicted_probability"].max(),
        calibration["observed_event_rate"].max(),
    )
    axis.plot([0, maximum], [0, maximum], color="#666666", linestyle="--", label="Ideal")
    axis.set_title("Validation Probability Calibration")
    axis.set_xlabel("Mean predicted bankruptcy probability")
    axis.set_ylabel("Observed bankruptcy rate")
    axis.set_xlim(0, maximum * 1.05)
    axis.set_ylim(0, maximum * 1.05)
    axis.legend(loc="upper left", fontsize=9)
    figure.tight_layout()
    return figure


def plot_selected_confusion_matrices(
    predictions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
) -> Figure:
    """Plot validation confusion matrices at each F2-selected threshold."""
    _apply_plot_style()
    threshold_map = selected_thresholds.set_index("model")["selected_threshold"].to_dict()
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, model_name in zip(axes.flat, SUBSTANTIVE_MODELS, strict=True):
        model_predictions = predictions.loc[predictions["model"].eq(model_name)]
        target = model_predictions[TARGET_COLUMN]
        predicted_class = model_predictions["predicted_probability"].ge(
            threshold_map[model_name]
        )
        matrix = confusion_matrix(target, predicted_class, labels=[0, 1])
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
            f"{MODEL_LABELS[model_name]}\nThreshold={threshold_map[model_name]:.4f}"
        )
        axis.set_xlabel("Predicted outcome")
        axis.set_ylabel("Observed outcome")
    figure.suptitle("Validation Confusion Matrices at F2-Selected Thresholds", y=1.02)
    figure.tight_layout()
    return figure
