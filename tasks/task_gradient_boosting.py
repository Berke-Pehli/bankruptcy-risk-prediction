"""Select, fit, and document the Gradient Boosting bankruptcy benchmark.

Inputs:
    - ``data/processed/model_dataset.parquet``
    - Feature, splitting, preprocessing, boosting, and configuration source modules.

Outputs:
    - ``outputs/models/gradient_boosting.joblib``
    - Tuning, selection, feature-importance, and validation-prediction tables.
    - ``outputs/figures/gradient_boosting_feature_importance.png``

Hyperparameters are selected exclusively within expanding folds of the training period.
The final ensemble is fitted on 1999-2011 observations before it scores validation data.
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
    FEATURES_MODULE_PATH,
    GRADIENT_BOOSTING_CV_RESULTS_PATH,
    GRADIENT_BOOSTING_FEATURE_IMPORTANCE_PATH,
    GRADIENT_BOOSTING_IMPORTANCE_FIGURE_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    GRADIENT_BOOSTING_MODULE_PATH,
    GRADIENT_BOOSTING_SELECTION_PATH,
    GRADIENT_BOOSTING_VALIDATION_PREDICTIONS_PATH,
    MODEL_DATASET_PATH,
    PREPROCESSING_MODULE_PATH,
    SPLITTING_MODULE_PATH,
)
from bankruptcy_risk.gradient_boosting import (
    build_gradient_boosting_feature_importance,
    create_gradient_boosting_validation_predictions,
    cross_validate_gradient_boosting,
    fit_selected_gradient_boosting,
    plot_gradient_boosting_feature_importance,
    select_gradient_boosting,
)


def task_fit_gradient_boosting(
    model_data_path: Path = MODEL_DATASET_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    features_module_path: Path = FEATURES_MODULE_PATH,
    splitting_module_path: Path = SPLITTING_MODULE_PATH,
    preprocessing_module_path: Path = PREPROCESSING_MODULE_PATH,
    boosting_module_path: Path = GRADIENT_BOOSTING_MODULE_PATH,
    model_path: Annotated[Path, Product] = GRADIENT_BOOSTING_MODEL_PATH,
    cv_results_path: Annotated[Path, Product] = GRADIENT_BOOSTING_CV_RESULTS_PATH,
    selection_path: Annotated[Path, Product] = GRADIENT_BOOSTING_SELECTION_PATH,
    importance_path: Annotated[
        Path, Product
    ] = GRADIENT_BOOSTING_FEATURE_IMPORTANCE_PATH,
    predictions_path: Annotated[
        Path, Product
    ] = GRADIENT_BOOSTING_VALIDATION_PREDICTIONS_PATH,
    figure_path: Annotated[Path, Product] = GRADIENT_BOOSTING_IMPORTANCE_FIGURE_PATH,
) -> None:
    """Tune boosting and write its model, tables, predictions, and figure."""
    source_paths = (
        config_module_path,
        features_module_path,
        splitting_module_path,
        preprocessing_module_path,
        boosting_module_path,
    )
    if not all(path.exists() for path in source_paths):
        raise FileNotFoundError("Gradient Boosting source modules must exist.")

    model_data = pd.read_parquet(model_data_path)
    cv_results = cross_validate_gradient_boosting(model_data)
    selection = select_gradient_boosting(cv_results)
    fitted_boosting = fit_selected_gradient_boosting(model_data, selection)
    importance = build_gradient_boosting_feature_importance(fitted_boosting)
    predictions = create_gradient_boosting_validation_predictions(
        model_data,
        fitted_boosting,
    )
    figure = plot_gradient_boosting_feature_importance(importance)

    output_paths = (
        model_path,
        cv_results_path,
        selection_path,
        importance_path,
        predictions_path,
        figure_path,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_boosting, model_path)
    cv_results.to_csv(cv_results_path, index=False)
    selection.to_csv(selection_path, index=False)
    importance.to_csv(importance_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
