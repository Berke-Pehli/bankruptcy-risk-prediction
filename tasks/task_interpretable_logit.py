"""Fit and document the interpretable Logistic Regression benchmark.

Inputs:
    - ``data/processed/model_dataset.parquet``
    - Feature, preprocessing, model, and configuration modules tracked by Pytask.

Outputs:
    - ``outputs/models/interpretable_logit.joblib``
    - ``outputs/tables/interpretable_logit_coefficients.csv``
    - ``outputs/tables/interpretable_logit_marginal_effects.csv``
    - ``outputs/tables/interpretable_logit_validation_predictions.csv``

The model and every preprocessing step are fitted only on 1999-2011 observations.
Validation predictions are generated afterward without refitting any component.
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
    INTERPRETABLE_LOGIT_COEFFICIENTS_PATH,
    INTERPRETABLE_LOGIT_MARGINAL_EFFECTS_PATH,
    INTERPRETABLE_LOGIT_MODEL_PATH,
    INTERPRETABLE_LOGIT_MODULE_PATH,
    INTERPRETABLE_LOGIT_VALIDATION_PREDICTIONS_PATH,
    MODEL_DATASET_PATH,
    PREPROCESSING_MODULE_PATH,
)
from bankruptcy_risk.interpretable_logit import (
    build_coefficient_table,
    build_marginal_effects_table,
    create_validation_logit_predictions,
    fit_interpretable_logit,
)


def task_fit_interpretable_logit(
    model_data_path: Path = MODEL_DATASET_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    features_module_path: Path = FEATURES_MODULE_PATH,
    preprocessing_module_path: Path = PREPROCESSING_MODULE_PATH,
    logit_module_path: Path = INTERPRETABLE_LOGIT_MODULE_PATH,
    model_path: Annotated[Path, Product] = INTERPRETABLE_LOGIT_MODEL_PATH,
    coefficient_path: Annotated[Path, Product] = INTERPRETABLE_LOGIT_COEFFICIENTS_PATH,
    marginal_effects_path: Annotated[Path, Product] = INTERPRETABLE_LOGIT_MARGINAL_EFFECTS_PATH,
    predictions_path: Annotated[Path, Product] = INTERPRETABLE_LOGIT_VALIDATION_PREDICTIONS_PATH,
) -> None:
    """Fit the training-only model and write its interpretation artifacts."""
    source_paths = (
        config_module_path,
        features_module_path,
        preprocessing_module_path,
        logit_module_path,
    )
    if not all(path.exists() for path in source_paths):
        raise FileNotFoundError("Interpretable-logit source modules must exist.")

    model_data = pd.read_parquet(model_data_path)
    fitted = fit_interpretable_logit(model_data)
    coefficients = build_coefficient_table(fitted)
    marginal_effects = build_marginal_effects_table(fitted)
    predictions = create_validation_logit_predictions(model_data, fitted)

    for path in (model_path, coefficient_path, marginal_effects_path, predictions_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, model_path)
    coefficients.to_csv(coefficient_path, index=False)
    marginal_effects.to_csv(marginal_effects_path, index=False)
    predictions.to_csv(predictions_path, index=False)
