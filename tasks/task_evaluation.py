"""Evaluate all bankruptcy models on the common validation period.

Inputs:
    - Validation prediction tables from every benchmark and fitted model.
    - Evaluation and configuration source modules tracked by Pytask.

Outputs:
    - Combined predictions, default metrics, threshold curves, selected thresholds,
      threshold-optimized metrics, and calibration-bin tables.
    - Precision-recall, ROC, calibration, and confusion-matrix figures.

This task uses 2012-2014 outcomes for model comparison and threshold selection only. It
does not access the reserved 2015-2018 final test observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product

from bankruptcy_risk.config import (
    COMBINED_VALIDATION_PREDICTIONS_PATH,
    CONFIG_MODULE_PATH,
    DECISION_TREE_VALIDATION_PREDICTIONS_PATH,
    EVALUATION_MODULE_PATH,
    GRADIENT_BOOSTING_VALIDATION_PREDICTIONS_PATH,
    INTERPRETABLE_LOGIT_VALIDATION_PREDICTIONS_PATH,
    RANDOM_FOREST_VALIDATION_PREDICTIONS_PATH,
    REGULARIZED_LOGIT_VALIDATION_PREDICTIONS_PATH,
    VALIDATION_BASELINE_PREDICTIONS_PATH,
    VALIDATION_CALIBRATION_FIGURE_PATH,
    VALIDATION_CALIBRATION_PATH,
    VALIDATION_CONFUSION_MATRICES_FIGURE_PATH,
    VALIDATION_DEFAULT_METRICS_PATH,
    VALIDATION_OPTIMIZED_METRICS_PATH,
    VALIDATION_PRECISION_RECALL_FIGURE_PATH,
    VALIDATION_ROC_FIGURE_PATH,
    VALIDATION_SELECTED_THRESHOLDS_PATH,
    VALIDATION_THRESHOLD_CURVES_PATH,
)
from bankruptcy_risk.evaluation import (
    SUBSTANTIVE_MODELS,
    build_calibration_table,
    build_threshold_curves,
    combine_validation_predictions,
    evaluate_predictions,
    plot_calibration_curves,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_selected_confusion_matrices,
    select_f2_thresholds,
)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one evaluation figure and release its Matplotlib resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def task_evaluate_validation_predictions(
    baseline_path: Path = VALIDATION_BASELINE_PREDICTIONS_PATH,
    interpretable_logit_path: Path = INTERPRETABLE_LOGIT_VALIDATION_PREDICTIONS_PATH,
    regularized_logit_path: Path = REGULARIZED_LOGIT_VALIDATION_PREDICTIONS_PATH,
    decision_tree_path: Path = DECISION_TREE_VALIDATION_PREDICTIONS_PATH,
    random_forest_path: Path = RANDOM_FOREST_VALIDATION_PREDICTIONS_PATH,
    gradient_boosting_path: Path = GRADIENT_BOOSTING_VALIDATION_PREDICTIONS_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    evaluation_module_path: Path = EVALUATION_MODULE_PATH,
    combined_path: Annotated[Path, Product] = COMBINED_VALIDATION_PREDICTIONS_PATH,
    default_metrics_path: Annotated[Path, Product] = VALIDATION_DEFAULT_METRICS_PATH,
    threshold_curves_path: Annotated[Path, Product] = VALIDATION_THRESHOLD_CURVES_PATH,
    selected_thresholds_path: Annotated[
        Path, Product
    ] = VALIDATION_SELECTED_THRESHOLDS_PATH,
    optimized_metrics_path: Annotated[
        Path, Product
    ] = VALIDATION_OPTIMIZED_METRICS_PATH,
    calibration_path: Annotated[Path, Product] = VALIDATION_CALIBRATION_PATH,
    precision_recall_figure_path: Annotated[
        Path, Product
    ] = VALIDATION_PRECISION_RECALL_FIGURE_PATH,
    roc_figure_path: Annotated[Path, Product] = VALIDATION_ROC_FIGURE_PATH,
    calibration_figure_path: Annotated[
        Path, Product
    ] = VALIDATION_CALIBRATION_FIGURE_PATH,
    confusion_figure_path: Annotated[
        Path, Product
    ] = VALIDATION_CONFUSION_MATRICES_FIGURE_PATH,
) -> None:
    """Write unified validation metrics, thresholds, calibration, and figures."""
    if not config_module_path.exists() or not evaluation_module_path.exists():
        raise FileNotFoundError("Evaluation source modules must exist.")
    prediction_paths = (
        baseline_path,
        interpretable_logit_path,
        regularized_logit_path,
        decision_tree_path,
        random_forest_path,
        gradient_boosting_path,
    )
    predictions = combine_validation_predictions(
        pd.read_csv(path) for path in prediction_paths
    )
    default_metrics = evaluate_predictions(predictions)
    threshold_curves = build_threshold_curves(predictions)
    selected_thresholds = select_f2_thresholds(threshold_curves)
    threshold_map = selected_thresholds.set_index("model")["selected_threshold"].to_dict()
    optimized_metrics = evaluate_predictions(
        predictions,
        thresholds=threshold_map,
        models=SUBSTANTIVE_MODELS,
        threshold_source="maximum_validation_f2",
    )
    calibration = build_calibration_table(predictions)

    table_outputs = (
        (predictions, combined_path),
        (default_metrics, default_metrics_path),
        (threshold_curves, threshold_curves_path),
        (selected_thresholds, selected_thresholds_path),
        (optimized_metrics, optimized_metrics_path),
        (calibration, calibration_path),
    )
    for table, path in table_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    figures = (
        (plot_precision_recall_curves(predictions), precision_recall_figure_path),
        (plot_roc_curves(predictions), roc_figure_path),
        (plot_calibration_curves(calibration), calibration_figure_path),
        (
            plot_selected_confusion_matrices(predictions, selected_thresholds),
            confusion_figure_path,
        ),
    )
    for figure, path in figures:
        _save_figure(figure, path)
