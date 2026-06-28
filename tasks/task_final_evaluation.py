"""Refit the frozen champion and evaluate the untouched final test period.

Inputs:
    - Validation metrics, thresholds, and Random Forest hyperparameter selection.
    - The complete temporally labelled model dataset.
    - Final-evaluation and configuration source modules tracked by Pytask.

Outputs:
    - Frozen specification, final model, test predictions, metrics, calibration,
      company-bootstrap intervals, validation-test comparison, and two figures.

All model and threshold choices are frozen before test outcomes enter any calculation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product

from bankruptcy_risk.config import (
    CONFIG_MODULE_PATH,
    FINAL_CHAMPION_MODEL_PATH,
    FINAL_EVALUATION_MODULE_PATH,
    FINAL_MODEL_SPECIFICATION_PATH,
    FINAL_TEST_BOOTSTRAP_INTERVALS_PATH,
    FINAL_TEST_BOOTSTRAP_REPLICATES_PATH,
    FINAL_TEST_CALIBRATION_PATH,
    FINAL_TEST_CONFUSION_MATRIX_FIGURE_PATH,
    FINAL_TEST_DIAGNOSTICS_FIGURE_PATH,
    FINAL_TEST_METRICS_PATH,
    FINAL_TEST_PREDICTIONS_PATH,
    MODEL_DATASET_PATH,
    RANDOM_FOREST_SELECTION_PATH,
    VALIDATION_DEFAULT_METRICS_PATH,
    VALIDATION_OPTIMIZED_METRICS_PATH,
    VALIDATION_SELECTED_THRESHOLDS_PATH,
    VALIDATION_TEST_COMPARISON_PATH,
)
from bankruptcy_risk.final_evaluation import (
    build_final_model_audit,
    build_final_test_calibration,
    build_final_test_intervals,
    build_validation_test_comparison,
    create_final_test_predictions,
    evaluate_final_test_predictions,
    fit_final_champion,
    freeze_final_specification,
    generate_final_test_bootstrap,
    plot_final_test_confusion_matrix,
    plot_final_test_diagnostics,
)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one final-evaluation figure and release its resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def task_evaluate_final_test_period(
    model_data_path: Path = MODEL_DATASET_PATH,
    validation_metrics_path: Path = VALIDATION_DEFAULT_METRICS_PATH,
    optimized_metrics_path: Path = VALIDATION_OPTIMIZED_METRICS_PATH,
    thresholds_path: Path = VALIDATION_SELECTED_THRESHOLDS_PATH,
    forest_selection_path: Path = RANDOM_FOREST_SELECTION_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    final_evaluation_module_path: Path = FINAL_EVALUATION_MODULE_PATH,
    model_path: Annotated[Path, Product] = FINAL_CHAMPION_MODEL_PATH,
    specification_path: Annotated[Path, Product] = FINAL_MODEL_SPECIFICATION_PATH,
    predictions_path: Annotated[Path, Product] = FINAL_TEST_PREDICTIONS_PATH,
    metrics_path: Annotated[Path, Product] = FINAL_TEST_METRICS_PATH,
    calibration_path: Annotated[Path, Product] = FINAL_TEST_CALIBRATION_PATH,
    bootstrap_path: Annotated[Path, Product] = FINAL_TEST_BOOTSTRAP_REPLICATES_PATH,
    intervals_path: Annotated[Path, Product] = FINAL_TEST_BOOTSTRAP_INTERVALS_PATH,
    comparison_path: Annotated[Path, Product] = VALIDATION_TEST_COMPARISON_PATH,
    diagnostics_figure_path: Annotated[
        Path, Product
    ] = FINAL_TEST_DIAGNOSTICS_FIGURE_PATH,
    confusion_figure_path: Annotated[
        Path, Product
    ] = FINAL_TEST_CONFUSION_MATRIX_FIGURE_PATH,
) -> None:
    """Fit the frozen champion and write one-time final-test evidence."""
    if not config_module_path.exists() or not final_evaluation_module_path.exists():
        raise FileNotFoundError("Final-evaluation source modules must exist.")
    model_data = pd.read_parquet(model_data_path)
    validation_metrics = pd.read_csv(validation_metrics_path)
    optimized_metrics = pd.read_csv(optimized_metrics_path)
    thresholds = pd.read_csv(thresholds_path)
    forest_selection = pd.read_csv(forest_selection_path)

    specification = freeze_final_specification(
        validation_metrics,
        thresholds,
        forest_selection,
    )
    fitted_model = fit_final_champion(model_data, specification)
    predictions = create_final_test_predictions(model_data, fitted_model, specification)
    metrics = evaluate_final_test_predictions(predictions, specification)
    audit = build_final_model_audit(model_data, specification)
    calibration = build_final_test_calibration(predictions)
    bootstrap = generate_final_test_bootstrap(predictions, specification)
    intervals = build_final_test_intervals(metrics, bootstrap)
    comparison = build_validation_test_comparison(optimized_metrics, metrics)

    output_paths = (
        model_path,
        specification_path,
        predictions_path,
        metrics_path,
        calibration_path,
        bootstrap_path,
        intervals_path,
        comparison_path,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_model, model_path)
    audit.to_csv(specification_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    intervals.to_csv(intervals_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    _save_figure(
        plot_final_test_diagnostics(predictions, calibration),
        diagnostics_figure_path,
    )
    _save_figure(
        plot_final_test_confusion_matrix(predictions, specification),
        confusion_figure_path,
    )
