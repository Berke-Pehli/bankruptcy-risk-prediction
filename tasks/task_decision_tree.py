"""Select, fit, and document the pruned bankruptcy decision tree.

Inputs:
    - ``data/processed/model_dataset.parquet``
    - Feature, splitting, preprocessing, tree, and configuration source modules.

Outputs:
    - ``outputs/models/pruned_decision_tree.joblib``
    - Cross-validation, selection, feature-importance, and validation-prediction tables.
    - ``outputs/figures/pruned_decision_tree.png``

All tuning occurs within expanding folds of the 1999-2011 training period. The final
tree scores validation observations only after its preprocessing and splits are fixed.
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
    DECISION_TREE_CV_RESULTS_PATH,
    DECISION_TREE_FEATURE_IMPORTANCE_PATH,
    DECISION_TREE_FIGURE_PATH,
    DECISION_TREE_MODEL_PATH,
    DECISION_TREE_MODULE_PATH,
    DECISION_TREE_SELECTION_PATH,
    DECISION_TREE_VALIDATION_PREDICTIONS_PATH,
    FEATURES_MODULE_PATH,
    MODEL_DATASET_PATH,
    PREPROCESSING_MODULE_PATH,
    SPLITTING_MODULE_PATH,
)
from bankruptcy_risk.decision_tree import (
    build_tree_feature_importance,
    create_tree_validation_predictions,
    cross_validate_pruned_tree,
    fit_selected_pruned_tree,
    plot_selected_tree,
    select_pruned_tree,
)


def task_fit_pruned_decision_tree(
    model_data_path: Path = MODEL_DATASET_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    features_module_path: Path = FEATURES_MODULE_PATH,
    splitting_module_path: Path = SPLITTING_MODULE_PATH,
    preprocessing_module_path: Path = PREPROCESSING_MODULE_PATH,
    tree_module_path: Path = DECISION_TREE_MODULE_PATH,
    model_path: Annotated[Path, Product] = DECISION_TREE_MODEL_PATH,
    cv_results_path: Annotated[Path, Product] = DECISION_TREE_CV_RESULTS_PATH,
    selection_path: Annotated[Path, Product] = DECISION_TREE_SELECTION_PATH,
    importance_path: Annotated[Path, Product] = DECISION_TREE_FEATURE_IMPORTANCE_PATH,
    predictions_path: Annotated[
        Path, Product
    ] = DECISION_TREE_VALIDATION_PREDICTIONS_PATH,
    figure_path: Annotated[Path, Product] = DECISION_TREE_FIGURE_PATH,
) -> None:
    """Tune the tree and write its model, tables, predictions, and figure."""
    source_paths = (
        config_module_path,
        features_module_path,
        splitting_module_path,
        preprocessing_module_path,
        tree_module_path,
    )
    if not all(path.exists() for path in source_paths):
        raise FileNotFoundError("Decision-tree source modules must exist.")

    model_data = pd.read_parquet(model_data_path)
    cv_results = cross_validate_pruned_tree(model_data)
    selection = select_pruned_tree(cv_results)
    fitted_tree = fit_selected_pruned_tree(model_data, selection)
    importance = build_tree_feature_importance(fitted_tree)
    predictions = create_tree_validation_predictions(model_data, fitted_tree)
    figure = plot_selected_tree(fitted_tree)

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
    joblib.dump(fitted_tree, model_path)
    cv_results.to_csv(cv_results_path, index=False)
    selection.to_csv(selection_path, index=False)
    importance.to_csv(importance_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
