"""Interpret the validation champion and compare financial drivers across models.

Inputs:
    - The temporal model dataset, fitted model artifacts, and validation metric table.
    - Interpretation and configuration source modules tracked by Pytask.

Outputs:
    - Champion permutation importance and partial-dependence tables.
    - Cross-model driver importance and consensus tables.
    - Three model-interpretation figures.

Validation outcomes are used for post-selection interpretation only. No final test-period
outcome is accessed, and all partial-dependence results are descriptive rather than causal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product

from bankruptcy_risk.config import (
    CHAMPION_PARTIAL_DEPENDENCE_FIGURE_PATH,
    CHAMPION_PARTIAL_DEPENDENCE_PATH,
    CHAMPION_PERMUTATION_IMPORTANCE_FIGURE_PATH,
    CHAMPION_PERMUTATION_IMPORTANCE_PATH,
    CONFIG_MODULE_PATH,
    CROSS_MODEL_DRIVER_CONSENSUS_PATH,
    CROSS_MODEL_DRIVER_HEATMAP_PATH,
    CROSS_MODEL_DRIVER_IMPORTANCE_PATH,
    DECISION_TREE_MODEL_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    INTERPRETABLE_LOGIT_MODEL_PATH,
    INTERPRETATION_MODULE_PATH,
    MODEL_DATASET_PATH,
    RANDOM_FOREST_MODEL_PATH,
    REGULARIZED_LOGIT_MODELS_PATH,
    VALIDATION_DEFAULT_METRICS_PATH,
)
from bankruptcy_risk.interpretation import (
    build_cross_model_driver_importance,
    build_driver_consensus,
    calculate_champion_partial_dependence,
    calculate_champion_permutation_importance,
    plot_champion_partial_dependence,
    plot_champion_permutation_importance,
    plot_cross_model_driver_heatmap,
    select_validation_champion,
)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one interpretation figure and release its Matplotlib resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def task_interpret_financial_drivers(
    model_data_path: Path = MODEL_DATASET_PATH,
    validation_metrics_path: Path = VALIDATION_DEFAULT_METRICS_PATH,
    interpretable_logit_path: Path = INTERPRETABLE_LOGIT_MODEL_PATH,
    regularized_logits_path: Path = REGULARIZED_LOGIT_MODELS_PATH,
    decision_tree_path: Path = DECISION_TREE_MODEL_PATH,
    random_forest_path: Path = RANDOM_FOREST_MODEL_PATH,
    gradient_boosting_path: Path = GRADIENT_BOOSTING_MODEL_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    interpretation_module_path: Path = INTERPRETATION_MODULE_PATH,
    permutation_path: Annotated[Path, Product] = CHAMPION_PERMUTATION_IMPORTANCE_PATH,
    partial_dependence_path: Annotated[
        Path, Product
    ] = CHAMPION_PARTIAL_DEPENDENCE_PATH,
    driver_importance_path: Annotated[
        Path, Product
    ] = CROSS_MODEL_DRIVER_IMPORTANCE_PATH,
    consensus_path: Annotated[Path, Product] = CROSS_MODEL_DRIVER_CONSENSUS_PATH,
    permutation_figure_path: Annotated[
        Path, Product
    ] = CHAMPION_PERMUTATION_IMPORTANCE_FIGURE_PATH,
    partial_dependence_figure_path: Annotated[
        Path, Product
    ] = CHAMPION_PARTIAL_DEPENDENCE_FIGURE_PATH,
    heatmap_path: Annotated[Path, Product] = CROSS_MODEL_DRIVER_HEATMAP_PATH,
) -> None:
    """Write model-behavior tables and financial-driver figures."""
    if not config_module_path.exists() or not interpretation_module_path.exists():
        raise FileNotFoundError("Interpretation source modules must exist.")

    model_data = pd.read_parquet(model_data_path)
    validation_metrics = pd.read_csv(validation_metrics_path)
    interpretable_logit = joblib.load(interpretable_logit_path)
    regularized_logits = joblib.load(regularized_logits_path)
    pruned_tree = joblib.load(decision_tree_path)
    random_forest = joblib.load(random_forest_path)
    gradient_boosting = joblib.load(gradient_boosting_path)
    fitted_models = {
        "ridge_logit": regularized_logits["ridge_logit"],
        "lasso_logit": regularized_logits["lasso_logit"],
        "pruned_decision_tree": pruned_tree,
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
    }

    champion = select_validation_champion(validation_metrics)
    if champion not in fitted_models:
        raise ValueError("The selected champion is not supported by the sklearn interpreter.")
    permutation = calculate_champion_permutation_importance(
        champion,
        fitted_models[champion],
        model_data,
    )
    partial_dependence_table = calculate_champion_partial_dependence(
        champion,
        fitted_models[champion],
        model_data,
        permutation,
    )
    driver_importance = build_cross_model_driver_importance(
        interpretable_logit,
        regularized_logits,
        pruned_tree,
        random_forest,
        gradient_boosting,
    )
    consensus = build_driver_consensus(driver_importance)

    table_outputs = (
        (permutation, permutation_path),
        (partial_dependence_table, partial_dependence_path),
        (driver_importance, driver_importance_path),
        (consensus, consensus_path),
    )
    for table, path in table_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    _save_figure(
        plot_champion_permutation_importance(permutation),
        permutation_figure_path,
    )
    _save_figure(
        plot_champion_partial_dependence(partial_dependence_table),
        partial_dependence_figure_path,
    )
    _save_figure(
        plot_cross_model_driver_heatmap(driver_importance, consensus),
        heatmap_path,
    )
