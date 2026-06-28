"""Select, fit, and explain a pruned bankruptcy classification tree.

This module introduces the first nonlinear model in the project. A classification tree
can represent threshold effects and interactions among financial indicators without
requiring those relationships to be specified in advance.

Inputs:
    - Training observations from ``data/processed/model_dataset.parquet``.
    - The expanding-window folds defined in ``splitting.py``.

Model selection:
    - Maximum depth and minimum leaf size provide pre-pruning.
    - ``ccp_alpha`` applies cost-complexity post-pruning.
    - PR-AUC is the primary score because bankruptcies are rare.
    - ROC-AUC is recorded as a complementary ranking measure.
    - The one-standard-error rule selects the smallest competitive tree.

Outputs:
    - Fold-level cross-validation results and the selected tree configuration.
    - A final model fitted only on 1999-2011 observations.
    - Validation probabilities, impurity-based feature importances, and a readable plot.

Every candidate contains its own median-imputation and clipping pipeline. Tree features
are not standardized because decision-tree splits are unaffected by feature scale. The
model remains unweighted so its leaf probabilities reflect the observed training event
rate; decision thresholds are addressed later in the evaluation stage.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from bankruptcy_risk.config import RANDOM_SEED
from bankruptcy_risk.features import MODEL_FEATURES
from bankruptcy_risk.preprocessing import make_tree_preprocessor
from bankruptcy_risk.splitting import EXPANDING_FOLDS, make_expanding_window_splits
from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN

DEFAULT_MAX_DEPTHS = (3, 5, 8)
DEFAULT_MIN_SAMPLES_LEAF = (25, 100)
DEFAULT_CCP_ALPHAS = (0.0, 0.000001, 0.00001, 0.0001)
MODEL_NAME = "pruned_decision_tree"


def _validate_model_data(model_data: pd.DataFrame) -> None:
    """Check that the temporal model table contains every tree input."""
    required_columns = {
        COMPANY_COLUMN,
        YEAR_COLUMN,
        TARGET_COLUMN,
        "sample_period",
        *MODEL_FEATURES,
    }
    missing_columns = required_columns.difference(model_data.columns)
    if missing_columns:
        raise ValueError(f"Required decision-tree columns are missing: {missing_columns}")


def make_decision_tree_pipeline(
    max_depth: int,
    min_samples_leaf: int,
    ccp_alpha: float,
) -> Pipeline:
    """Create a leakage-safe preprocessing and classification-tree pipeline."""
    if max_depth < 1:
        raise ValueError("Maximum tree depth must be positive.")
    if min_samples_leaf < 1:
        raise ValueError("Minimum leaf size must be positive.")
    if ccp_alpha < 0:
        raise ValueError("Cost-complexity alpha cannot be negative.")

    return Pipeline(
        steps=[
            ("preprocessor", make_tree_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    ccp_alpha=ccp_alpha,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def cross_validate_pruned_tree(
    model_data: pd.DataFrame,
    max_depths: Iterable[int] = DEFAULT_MAX_DEPTHS,
    min_samples_leaf_values: Iterable[int] = DEFAULT_MIN_SAMPLES_LEAF,
    ccp_alphas: Iterable[float] = DEFAULT_CCP_ALPHAS,
) -> pd.DataFrame:
    """Evaluate candidate trees using expanding-window cross-validation.

    Imputation, clipping, and tree estimation are repeated inside every fold. The fold's
    validation years therefore cannot influence preprocessing boundaries or tree splits.
    """
    _validate_model_data(model_data)
    training = model_data.loc[model_data["sample_period"].eq("train")].reset_index(
        drop=True
    )
    if training.empty:
        raise ValueError("Training observations are required for tree selection.")

    depth_values = tuple(int(value) for value in max_depths)
    leaf_values = tuple(int(value) for value in min_samples_leaf_values)
    alpha_values = tuple(float(value) for value in ccp_alphas)
    if not depth_values or any(value < 1 for value in depth_values):
        raise ValueError("At least one positive maximum depth is required.")
    if not leaf_values or any(value < 1 for value in leaf_values):
        raise ValueError("At least one positive minimum leaf size is required.")
    if not alpha_values or any(value < 0 for value in alpha_values):
        raise ValueError("At least one non-negative cost-complexity alpha is required.")

    features = training.loc[:, MODEL_FEATURES]
    target = training[TARGET_COLUMN]
    splits = make_expanding_window_splits(training[YEAR_COLUMN])
    rows: list[dict[str, float | int | str]] = []

    for max_depth in depth_values:
        for min_samples_leaf in leaf_values:
            for ccp_alpha in alpha_values:
                for fold, (train_indices, validation_indices) in zip(
                    EXPANDING_FOLDS,
                    splits,
                    strict=True,
                ):
                    pipeline = make_decision_tree_pipeline(
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        ccp_alpha=ccp_alpha,
                    )
                    pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
                    probabilities = pipeline.predict_proba(
                        features.iloc[validation_indices]
                    )[:, 1]
                    validation_target = target.iloc[validation_indices]
                    tree = pipeline.named_steps["classifier"].tree_

                    rows.append(
                        {
                            "model": MODEL_NAME,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "ccp_alpha": ccp_alpha,
                            "fold": fold.name,
                            "train_end_year": fold.train_end,
                            "validation_start_year": fold.validation_start,
                            "validation_end_year": fold.validation_end,
                            "train_observations": len(train_indices),
                            "validation_observations": len(validation_indices),
                            "validation_bankruptcies": int(validation_target.sum()),
                            "fitted_tree_depth": tree.max_depth,
                            "fitted_leaf_count": tree.n_leaves,
                            "pr_auc": average_precision_score(
                                validation_target,
                                probabilities,
                            ),
                            "roc_auc": roc_auc_score(
                                validation_target,
                                probabilities,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def select_pruned_tree(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Choose the simplest competitive tree using the one-standard-error rule.

    The best candidate defines an eligibility threshold equal to its mean PR-AUC minus
    one standard error. Among candidates above that threshold, the tree with the fewest
    average leaves is selected. Average fitted depth and stronger pruning break ties.
    """
    required_columns = {
        "model",
        "max_depth",
        "min_samples_leaf",
        "ccp_alpha",
        "fitted_tree_depth",
        "fitted_leaf_count",
        "pr_auc",
        "roc_auc",
    }
    missing_columns = required_columns.difference(cv_results.columns)
    if missing_columns:
        raise ValueError(f"Tree cross-validation columns are missing: {missing_columns}")

    summary = (
        cv_results.groupby(
            ["model", "max_depth", "min_samples_leaf", "ccp_alpha"],
            as_index=False,
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_fitted_depth=("fitted_tree_depth", "mean"),
            mean_fitted_leaf_count=("fitted_leaf_count", "mean"),
            folds=("pr_auc", "size"),
        )
    )
    best = summary.loc[summary["mean_pr_auc"].idxmax()]
    standard_error = best["std_pr_auc"] / np.sqrt(best["folds"])
    eligibility_threshold = best["mean_pr_auc"] - standard_error
    eligible = summary.loc[summary["mean_pr_auc"].ge(eligibility_threshold)].copy()
    selected = eligible.sort_values(
        [
            "mean_fitted_leaf_count",
            "mean_fitted_depth",
            "ccp_alpha",
            "max_depth",
            "min_samples_leaf",
        ],
        ascending=[True, True, False, True, False],
    ).head(1)
    selected["best_mean_pr_auc"] = best["mean_pr_auc"]
    selected["one_standard_error"] = standard_error
    selected["eligibility_threshold"] = eligibility_threshold
    selected["selection_rule"] = "one_standard_error_simplest_tree"
    return selected.reset_index(drop=True)


def fit_selected_pruned_tree(
    model_data: pd.DataFrame,
    selection: pd.DataFrame,
) -> Pipeline:
    """Fit the selected tree pipeline on the complete 1999-2011 training period."""
    _validate_model_data(model_data)
    if len(selection) != 1:
        raise ValueError("Tree selection must contain exactly one candidate.")
    training = model_data.loc[model_data["sample_period"].eq("train")]
    if training.empty:
        raise ValueError("Training observations are required to fit the final tree.")

    selected = selection.iloc[0]
    pipeline = make_decision_tree_pipeline(
        max_depth=int(selected["max_depth"]),
        min_samples_leaf=int(selected["min_samples_leaf"]),
        ccp_alpha=float(selected["ccp_alpha"]),
    )
    pipeline.fit(training.loc[:, MODEL_FEATURES], training[TARGET_COLUMN])
    return pipeline


def build_tree_feature_importance(fitted_tree: Pipeline) -> pd.DataFrame:
    """Return ranked impurity-based feature importances for the selected tree."""
    classifier = fitted_tree.named_steps["classifier"]
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance["used_in_tree"] = importance["importance"].gt(0)
    return importance.loc[:, ["rank", "feature", "importance", "used_in_tree"]]


def create_tree_validation_predictions(
    model_data: pd.DataFrame,
    fitted_tree: Pipeline,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create bankruptcy probabilities for the 2012-2014 validation period."""
    _validate_model_data(model_data)
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must lie between zero and one.")
    validation = model_data.loc[model_data["sample_period"].eq("validation")]
    if validation.empty:
        raise ValueError("Validation observations are required to create predictions.")

    probabilities = fitted_tree.predict_proba(validation.loc[:, MODEL_FEATURES])[:, 1]
    predictions = validation.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN]].copy()
    predictions["model"] = MODEL_NAME
    predictions["predicted_probability"] = probabilities
    predictions["predicted_class"] = (probabilities >= threshold).astype("int8")
    return predictions.reset_index(drop=True)


def plot_selected_tree(fitted_tree: Pipeline, displayed_depth: int = 3) -> Figure:
    """Plot the first levels of the selected tree for readable financial interpretation."""
    if displayed_depth < 1:
        raise ValueError("Displayed tree depth must be positive.")
    classifier = fitted_tree.named_steps["classifier"]
    figure, axis = plt.subplots(figsize=(20, 11))
    plot_tree(
        classifier,
        feature_names=list(MODEL_FEATURES),
        class_names=["No bankruptcy", "Bankruptcy next year"],
        filled=True,
        rounded=True,
        proportion=True,
        impurity=False,
        max_depth=displayed_depth,
        fontsize=7,
        ax=axis,
    )
    axis.set_title(
        f"Cost-Complexity-Pruned Decision Tree (First {displayed_depth + 1} Levels)"
    )
    figure.tight_layout()
    return figure
