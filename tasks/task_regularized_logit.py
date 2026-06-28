"""Select, fit, and document regularized Logistic Regression models.

Inputs:
    - ``data/processed/model_dataset.parquet``
    - Feature, splitting, preprocessing, model, and configuration source modules.

Outputs:
    - ``outputs/models/regularized_logit_models.joblib``
    - ``outputs/tables/regularized_logit_cv_results.csv``
    - ``outputs/tables/regularized_logit_selection.csv``
    - ``outputs/tables/regularized_logit_coefficients.csv``
    - ``outputs/tables/regularized_logit_validation_predictions.csv``

Hyperparameters are selected exclusively within expanding folds of the 1999-2011
training period. The 2012-2014 validation observations are scored only after both final
models and their preprocessing pipelines have been fitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from pytask import Product

from bankruptcy_risk.config import (
    CONFIG_MODULE_PATH,
    FEATURES_MODULE_PATH,
    MODEL_DATASET_PATH,
    PREPROCESSING_MODULE_PATH,
    REGULARIZED_LOGIT_COEFFICIENTS_PATH,
    REGULARIZED_LOGIT_CV_RESULTS_PATH,
    REGULARIZED_LOGIT_MODELS_PATH,
    REGULARIZED_LOGIT_MODULE_PATH,
    REGULARIZED_LOGIT_SELECTION_PATH,
    REGULARIZED_LOGIT_VALIDATION_PREDICTIONS_PATH,
    SPLITTING_MODULE_PATH,
)
from bankruptcy_risk.regularized_logit import (
    build_regularized_coefficient_table,
    create_regularized_validation_predictions,
    cross_validate_regularized_logits,
    fit_selected_regularized_logits,
    select_regularization_strengths,
)


def task_fit_regularized_logits(
    model_data_path: Path = MODEL_DATASET_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    features_module_path: Path = FEATURES_MODULE_PATH,
    splitting_module_path: Path = SPLITTING_MODULE_PATH,
    preprocessing_module_path: Path = PREPROCESSING_MODULE_PATH,
    model_module_path: Path = REGULARIZED_LOGIT_MODULE_PATH,
    models_path: Annotated[Path, Product] = REGULARIZED_LOGIT_MODELS_PATH,
    cv_results_path: Annotated[Path, Product] = REGULARIZED_LOGIT_CV_RESULTS_PATH,
    selection_path: Annotated[Path, Product] = REGULARIZED_LOGIT_SELECTION_PATH,
    coefficients_path: Annotated[Path, Product] = REGULARIZED_LOGIT_COEFFICIENTS_PATH,
    predictions_path: Annotated[
        Path, Product
    ] = REGULARIZED_LOGIT_VALIDATION_PREDICTIONS_PATH,
) -> None:
    """Tune on expanding folds and write final model artifacts."""
    source_paths = (
        config_module_path,
        features_module_path,
        splitting_module_path,
        preprocessing_module_path,
        model_module_path,
    )
    if not all(path.exists() for path in source_paths):
        raise FileNotFoundError("Regularized-logit source modules must exist.")

    model_data = pd.read_parquet(model_data_path)
    cv_results = cross_validate_regularized_logits(model_data)
    selection = select_regularization_strengths(cv_results)
    fitted_models = fit_selected_regularized_logits(model_data, selection)
    coefficients = build_regularized_coefficient_table(fitted_models)
    predictions = create_regularized_validation_predictions(model_data, fitted_models)

    output_paths = (
        models_path,
        cv_results_path,
        selection_path,
        coefficients_path,
        predictions_path,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models, models_path)
    cv_results.to_csv(cv_results_path, index=False)
    selection.to_csv(selection_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    predictions.to_csv(predictions_path, index=False)
