"""Select, fit, and document the Random Forest bankruptcy benchmark.

Inputs:
    - ``data/processed/model_dataset.parquet``
    - Feature, splitting, preprocessing, forest, and configuration source modules.

Outputs:
    - ``outputs/models/random_forest.joblib``
    - Tuning, selection, out-of-bag, importance, and validation-prediction tables.
    - ``outputs/figures/random_forest_feature_importance.png``

Hyperparameters are selected only within expanding folds of the training period. The
final forest is then fitted on 1999-2011 data and evaluated out of bag before it scores
the untouched 2012-2014 validation observations.
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
    MODEL_DATASET_PATH,
    PREPROCESSING_MODULE_PATH,
    RANDOM_FOREST_CV_RESULTS_PATH,
    RANDOM_FOREST_FEATURE_IMPORTANCE_PATH,
    RANDOM_FOREST_IMPORTANCE_FIGURE_PATH,
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_FOREST_MODULE_PATH,
    RANDOM_FOREST_OOB_DIAGNOSTICS_PATH,
    RANDOM_FOREST_SELECTION_PATH,
    RANDOM_FOREST_VALIDATION_PREDICTIONS_PATH,
    SPLITTING_MODULE_PATH,
)
from bankruptcy_risk.random_forest import (
    build_random_forest_feature_importance,
    build_random_forest_oob_diagnostics,
    create_random_forest_validation_predictions,
    cross_validate_random_forest,
    fit_selected_random_forest,
    plot_random_forest_feature_importance,
    select_random_forest,
)


def task_fit_random_forest(
    model_data_path: Path = MODEL_DATASET_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    features_module_path: Path = FEATURES_MODULE_PATH,
    splitting_module_path: Path = SPLITTING_MODULE_PATH,
    preprocessing_module_path: Path = PREPROCESSING_MODULE_PATH,
    forest_module_path: Path = RANDOM_FOREST_MODULE_PATH,
    model_path: Annotated[Path, Product] = RANDOM_FOREST_MODEL_PATH,
    cv_results_path: Annotated[Path, Product] = RANDOM_FOREST_CV_RESULTS_PATH,
    selection_path: Annotated[Path, Product] = RANDOM_FOREST_SELECTION_PATH,
    oob_path: Annotated[Path, Product] = RANDOM_FOREST_OOB_DIAGNOSTICS_PATH,
    importance_path: Annotated[Path, Product] = RANDOM_FOREST_FEATURE_IMPORTANCE_PATH,
    predictions_path: Annotated[
        Path, Product
    ] = RANDOM_FOREST_VALIDATION_PREDICTIONS_PATH,
    figure_path: Annotated[Path, Product] = RANDOM_FOREST_IMPORTANCE_FIGURE_PATH,
) -> None:
    """Tune the forest and write its model, diagnostics, predictions, and figure."""
    source_paths = (
        config_module_path,
        features_module_path,
        splitting_module_path,
        preprocessing_module_path,
        forest_module_path,
    )
    if not all(path.exists() for path in source_paths):
        raise FileNotFoundError("Random Forest source modules must exist.")

    model_data = pd.read_parquet(model_data_path)
    cv_results = cross_validate_random_forest(model_data)
    selection = select_random_forest(cv_results)
    fitted_forest = fit_selected_random_forest(model_data, selection)
    oob_diagnostics = build_random_forest_oob_diagnostics(model_data, fitted_forest)
    importance = build_random_forest_feature_importance(fitted_forest)
    predictions = create_random_forest_validation_predictions(model_data, fitted_forest)
    figure = plot_random_forest_feature_importance(importance)

    output_paths = (
        model_path,
        cv_results_path,
        selection_path,
        oob_path,
        importance_path,
        predictions_path,
        figure_path,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_forest, model_path)
    cv_results.to_csv(cv_results_path, index=False)
    selection.to_csv(selection_path, index=False)
    oob_diagnostics.to_csv(oob_path, index=False)
    importance.to_csv(importance_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
