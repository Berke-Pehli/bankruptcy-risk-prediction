"""Interpret the validation champion and compare financial drivers across models.

This module complements predictive metrics with model-behavior diagnostics. It first
selects the validation champion by PR-AUC and then measures how much its ranking quality
declines when each predictor is permuted.

Inputs:
    - Fitted models and the temporally labelled model dataset.
    - Unified validation metrics used to identify the champion.

Outputs:
    - Repeated validation permutation importance for the champion model.
    - Partial-dependence profiles for its four leading permutation features.
    - Normalized within-model importance for all six substantive models.
    - A coverage-aware consensus ranking of financial drivers.
    - Three portfolio-ready interpretation figures.

Permutation importance is model-agnostic and scored with PR-AUC. Partial dependence
describes average model predictions as one feature varies, but it does not identify
causal effects and can be misleading when financial predictors are strongly correlated.
Coefficient magnitudes and impurity importances are normalized only within their source
model; they should be interpreted as relative emphasis rather than a common effect scale.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.inspection import partial_dependence, permutation_importance

from bankruptcy_risk.config import RANDOM_SEED
from bankruptcy_risk.evaluation import MODEL_LABELS, SUBSTANTIVE_MODELS
from bankruptcy_risk.features import INTERPRETABLE_FEATURES, MODEL_FEATURES
from bankruptcy_risk.target import TARGET_COLUMN

PERMUTATION_REPEATS = 10
PERMUTATION_MAX_SAMPLES = 5_000
PARTIAL_DEPENDENCE_FEATURES = 4
PARTIAL_DEPENDENCE_GRID_POINTS = 30


def select_validation_champion(validation_metrics: pd.DataFrame) -> str:
    """Return the substantive model with the highest validation PR-AUC."""
    required_columns = {"model", "pr_auc"}
    missing_columns = required_columns.difference(validation_metrics.columns)
    if missing_columns:
        raise ValueError(f"Validation-metric columns are missing: {missing_columns}")
    substantive = validation_metrics.loc[
        validation_metrics["model"].isin(SUBSTANTIVE_MODELS)
    ]
    if set(substantive["model"]) != set(SUBSTANTIVE_MODELS):
        raise ValueError("Validation metrics must cover every substantive model.")
    ranking = substantive.sort_values(
        ["pr_auc", "model"],
        ascending=[False, True],
    )
    return str(ranking.iloc[0]["model"])


def calculate_champion_permutation_importance(
    champion_model: str,
    fitted_model: Any,
    model_data: pd.DataFrame,
    n_repeats: int = PERMUTATION_REPEATS,
    max_samples: int = PERMUTATION_MAX_SAMPLES,
) -> pd.DataFrame:
    """Measure validation PR-AUC reductions after repeated feature permutations."""
    if champion_model not in SUBSTANTIVE_MODELS:
        raise ValueError("Champion must be one of the substantive models.")
    if n_repeats < 1 or max_samples < 1:
        raise ValueError("Permutation repeats and maximum samples must be positive.")
    validation = model_data.loc[model_data["sample_period"].eq("validation")]
    if validation.empty:
        raise ValueError("Validation observations are required for interpretation.")

    sampled_observations = min(max_samples, len(validation))
    interpretation_sample = validation.sample(
        n=sampled_observations,
        random_state=RANDOM_SEED,
    )
    features = interpretation_sample.loc[:, MODEL_FEATURES]
    target = interpretation_sample[TARGET_COLUMN]
    result = permutation_importance(
        fitted_model,
        features,
        target,
        scoring="average_precision",
        n_repeats=n_repeats,
        n_jobs=1,
        random_state=RANDOM_SEED,
    )
    importance = pd.DataFrame(
        {
            "model": champion_model,
            "model_label": MODEL_LABELS[champion_model],
            "feature": MODEL_FEATURES,
            "mean_pr_auc_decrease": result.importances_mean,
            "std_pr_auc_decrease": result.importances_std,
            "permutation_repeats": n_repeats,
            "observations_per_repeat": sampled_observations,
            "scoring_metric": "average_precision",
        }
    ).sort_values("mean_pr_auc_decrease", ascending=False, ignore_index=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance.loc[
        :,
        [
            "rank",
            "model",
            "model_label",
            "feature",
            "mean_pr_auc_decrease",
            "std_pr_auc_decrease",
            "permutation_repeats",
            "observations_per_repeat",
            "scoring_metric",
        ],
    ]


def calculate_champion_partial_dependence(
    champion_model: str,
    fitted_model: Any,
    model_data: pd.DataFrame,
    permutation_importance_table: pd.DataFrame,
    n_features: int = PARTIAL_DEPENDENCE_FEATURES,
    grid_resolution: int = PARTIAL_DEPENDENCE_GRID_POINTS,
    max_samples: int = PERMUTATION_MAX_SAMPLES,
) -> pd.DataFrame:
    """Calculate average partial dependence for leading champion predictors."""
    if n_features < 1 or grid_resolution < 2 or max_samples < 1:
        raise ValueError("Partial-dependence settings must be positive and non-trivial.")
    validation = model_data.loc[model_data["sample_period"].eq("validation")]
    if validation.empty:
        raise ValueError("Validation observations are required for partial dependence.")
    interpretation_sample = validation.loc[:, MODEL_FEATURES].sample(
        n=min(max_samples, len(validation)),
        random_state=RANDOM_SEED,
    )
    leading_features = permutation_importance_table.head(n_features)["feature"]
    rows = []
    for feature in leading_features:
        result = partial_dependence(
            fitted_model,
            interpretation_sample,
            features=[feature],
            response_method="predict_proba",
            percentiles=(0.05, 0.95),
            grid_resolution=grid_resolution,
            kind="average",
        )
        grid_values = result["grid_values"][0]
        average_predictions = result["average"].ravel()
        rows.append(
            pd.DataFrame(
                {
                    "model": champion_model,
                    "model_label": MODEL_LABELS[champion_model],
                    "feature": feature,
                    "grid_value": grid_values,
                    "average_predicted_probability": average_predictions,
                    "interpretation_sample_size": len(interpretation_sample),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _append_normalized_importance(
    rows: list[pd.DataFrame],
    model_name: str,
    features: tuple[str, ...],
    values: np.ndarray,
    measure: str,
) -> None:
    """Append one model's ranked and normalized importance to a shared list."""
    values = np.asarray(values, dtype=float)
    if len(features) != len(values) or np.any(values < 0):
        raise ValueError(f"Importance values are invalid for {model_name}.")
    total = values.sum()
    normalized = values / total if total > 0 else np.zeros_like(values)
    table = pd.DataFrame(
        {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "feature": features,
            "importance_measure": measure,
            "raw_importance": values,
            "normalized_importance": normalized,
        }
    ).sort_values("normalized_importance", ascending=False, ignore_index=True)
    table["within_model_rank"] = np.arange(1, len(table) + 1)
    rows.append(table)


def build_cross_model_driver_importance(
    interpretable_logit: Any,
    regularized_logits: Mapping[str, Any],
    pruned_tree: Any,
    random_forest: Any,
    gradient_boosting: Any,
) -> pd.DataFrame:
    """Create within-model normalized financial-driver importance for six models."""
    rows: list[pd.DataFrame] = []
    interpretable_coefficients = interpretable_logit.result.params.loc[
        list(INTERPRETABLE_FEATURES)
    ].abs()
    _append_normalized_importance(
        rows,
        "interpretable_logit",
        INTERPRETABLE_FEATURES,
        interpretable_coefficients.to_numpy(),
        "absolute_standardized_coefficient",
    )
    for model_name in ("ridge_logit", "lasso_logit"):
        coefficients = np.abs(
            regularized_logits[model_name].named_steps["classifier"].coef_[0]
        )
        _append_normalized_importance(
            rows,
            model_name,
            MODEL_FEATURES,
            coefficients,
            "absolute_standardized_coefficient",
        )
    for model_name, model in (
        ("pruned_decision_tree", pruned_tree),
        ("random_forest", random_forest),
        ("gradient_boosting", gradient_boosting),
    ):
        _append_normalized_importance(
            rows,
            model_name,
            MODEL_FEATURES,
            model.named_steps["classifier"].feature_importances_,
            "mean_decrease_in_impurity",
        )
    return pd.concat(rows, ignore_index=True)


def build_driver_consensus(driver_importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized importance with explicit cross-model coverage."""
    required_columns = {
        "model",
        "feature",
        "normalized_importance",
        "within_model_rank",
    }
    missing_columns = required_columns.difference(driver_importance.columns)
    if missing_columns:
        raise ValueError(f"Driver-importance columns are missing: {missing_columns}")
    consensus = (
        driver_importance.groupby("feature", as_index=False)
        .agg(
            models_available=("model", "nunique"),
            mean_normalized_importance=("normalized_importance", "mean"),
            median_within_model_rank=("within_model_rank", "median"),
            top_five_appearances=("within_model_rank", lambda values: int((values <= 5).sum())),
        )
    )
    consensus["model_coverage_fraction"] = consensus["models_available"] / len(
        SUBSTANTIVE_MODELS
    )
    consensus["coverage_adjusted_importance"] = (
        consensus["mean_normalized_importance"] * consensus["model_coverage_fraction"]
    )
    consensus = consensus.sort_values(
        ["coverage_adjusted_importance", "top_five_appearances", "feature"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    consensus["consensus_rank"] = np.arange(1, len(consensus) + 1)
    return consensus.loc[
        :,
        [
            "consensus_rank",
            "feature",
            "models_available",
            "model_coverage_fraction",
            "mean_normalized_importance",
            "coverage_adjusted_importance",
            "median_within_model_rank",
            "top_five_appearances",
        ],
    ]


def plot_champion_permutation_importance(
    importance: pd.DataFrame,
    top_n: int = 15,
) -> Figure:
    """Plot the champion's largest validation PR-AUC permutation decreases."""
    if top_n < 1:
        raise ValueError("At least one permutation feature must be displayed.")
    sns.set_theme(style="whitegrid", context="notebook")
    displayed = importance.head(top_n).sort_values("mean_pr_auc_decrease", ascending=True)
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.barh(
        displayed["feature"].str.replace("_", " ").str.title(),
        displayed["mean_pr_auc_decrease"],
        xerr=displayed["std_pr_auc_decrease"],
        color="#1F4E79",
        alpha=0.85,
        error_kw={"ecolor": "#666666", "capsize": 2, "elinewidth": 0.8},
    )
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_title(f"{importance.iloc[0]['model_label']} Permutation Importance")
    axis.set_xlabel("Validation PR-AUC decrease after permutation")
    axis.set_ylabel("")
    figure.tight_layout()
    return figure


def plot_champion_partial_dependence(partial_dependence_table: pd.DataFrame) -> Figure:
    """Plot average partial-dependence profiles for four leading features."""
    sns.set_theme(style="whitegrid", context="notebook")
    features = partial_dependence_table["feature"].drop_duplicates().tolist()
    if len(features) != PARTIAL_DEPENDENCE_FEATURES:
        raise ValueError("The partial-dependence figure requires exactly four features.")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, feature in zip(axes.flat, features, strict=True):
        values = partial_dependence_table.loc[
            partial_dependence_table["feature"].eq(feature)
        ]
        axis.plot(
            values["grid_value"],
            values["average_predicted_probability"],
            color="#C55A11",
            linewidth=2,
        )
        axis.set_title(feature.replace("_", " ").title())
        axis.set_xlabel("Feature value")
        axis.set_ylabel("Average predicted probability")
    model_label = partial_dependence_table.iloc[0]["model_label"]
    figure.suptitle(f"{model_label} Partial Dependence", y=1.02)
    figure.tight_layout()
    return figure


def plot_cross_model_driver_heatmap(
    driver_importance: pd.DataFrame,
    consensus: pd.DataFrame,
    top_n: int = 15,
) -> Figure:
    """Plot normalized importance for the highest-ranked consensus drivers."""
    if top_n < 1:
        raise ValueError("At least one consensus feature must be displayed.")
    leading_features = consensus.head(top_n)["feature"]
    heatmap_data = driver_importance.loc[
        driver_importance["feature"].isin(leading_features)
    ].pivot(index="feature", columns="model", values="normalized_importance")
    heatmap_data = heatmap_data.reindex(index=leading_features)
    heatmap_data = heatmap_data.reindex(columns=SUBSTANTIVE_MODELS)
    heatmap_data.columns = [MODEL_LABELS[model] for model in heatmap_data.columns]

    sns.set_theme(style="white", context="notebook")
    figure, axis = plt.subplots(figsize=(13, 9))
    sns.heatmap(
        heatmap_data,
        mask=heatmap_data.isna(),
        cmap="Blues",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Within-model normalized importance"},
        ax=axis,
    )
    axis.set_title("Cross-Model Financial-Driver Comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("Financial predictor")
    axis.set_yticklabels(
        [label.get_text().replace("_", " ").title() for label in axis.get_yticklabels()],
        rotation=0,
    )
    figure.tight_layout()
    return figure
