"""Select, fit, and diagnose a Random Forest bankruptcy classifier.

Random Forests average many decorrelated classification trees. Bootstrap samples reduce
the instability of an individual tree, while random feature subsets allow different
financial signals to contribute across the ensemble.

Inputs:
    - Training observations from ``data/processed/model_dataset.parquet``.
    - The expanding-window folds defined in ``splitting.py``.

Model selection:
    - Maximum tree depth controls individual-tree complexity.
    - Minimum leaf size smooths leaf-level bankruptcy probabilities.
    - ``max_features`` controls the random subset considered at each split.
    - PR-AUC is the primary score and ROC-AUC is recorded for comparison.
    - The one-standard-error rule favors a simpler competitive forest.

Outputs:
    - Fold-level tuning results and the selected hyperparameters.
    - A final 500-tree model fitted on 1999-2011 observations only.
    - Out-of-bag PR-AUC, ROC-AUC, and Brier-score diagnostics.
    - Validation probabilities and impurity-based feature importances.

Each candidate pipeline learns imputation and clipping within its own training fold.
Features are not standardized because tree splits are scale-invariant. The forest is
unweighted so its probabilities retain the observed event prevalence; threshold choice
and calibration are evaluated explicitly in later milestones.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from bankruptcy_risk.config import RANDOM_SEED
from bankruptcy_risk.features import MODEL_FEATURES
from bankruptcy_risk.preprocessing import make_tree_preprocessor
from bankruptcy_risk.splitting import EXPANDING_FOLDS, make_expanding_window_splits
from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN

DEFAULT_MAX_DEPTHS = (8, 16)
DEFAULT_MIN_SAMPLES_LEAF = (25, 100)
DEFAULT_MAX_FEATURES = (0.2, 0.5)
TUNING_ESTIMATORS = 200
FINAL_ESTIMATORS = 500
MODEL_NAME = "random_forest"


def _validate_model_data(model_data: pd.DataFrame) -> None:
    """Check that the temporal model table contains every forest input."""
    required_columns = {
        COMPANY_COLUMN,
        YEAR_COLUMN,
        TARGET_COLUMN,
        "sample_period",
        *MODEL_FEATURES,
    }
    missing_columns = required_columns.difference(model_data.columns)
    if missing_columns:
        raise ValueError(f"Required Random Forest columns are missing: {missing_columns}")


def make_random_forest_pipeline(
    max_depth: int,
    min_samples_leaf: int,
    max_features: float,
    n_estimators: int,
    *,
    oob_score: bool = False,
) -> Pipeline:
    """Create a leakage-safe preprocessing and Random Forest pipeline."""
    if max_depth < 1:
        raise ValueError("Maximum tree depth must be positive.")
    if min_samples_leaf < 1:
        raise ValueError("Minimum leaf size must be positive.")
    if not 0 < max_features <= 1:
        raise ValueError("Maximum feature fraction must lie in (0, 1].")
    if n_estimators < 1:
        raise ValueError("The forest must contain at least one tree.")

    return Pipeline(
        steps=[
            ("preprocessor", make_tree_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=True,
                    oob_score=oob_score,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def cross_validate_random_forest(
    model_data: pd.DataFrame,
    max_depths: Iterable[int] = DEFAULT_MAX_DEPTHS,
    min_samples_leaf_values: Iterable[int] = DEFAULT_MIN_SAMPLES_LEAF,
    max_features_values: Iterable[float] = DEFAULT_MAX_FEATURES,
    n_estimators: int = TUNING_ESTIMATORS,
) -> pd.DataFrame:
    """Evaluate Random Forest candidates with expanding-window validation."""
    _validate_model_data(model_data)
    training = model_data.loc[model_data["sample_period"].eq("train")].reset_index(
        drop=True
    )
    if training.empty:
        raise ValueError("Training observations are required for forest selection.")

    depth_values = tuple(int(value) for value in max_depths)
    leaf_values = tuple(int(value) for value in min_samples_leaf_values)
    feature_values = tuple(float(value) for value in max_features_values)
    if not depth_values or any(value < 1 for value in depth_values):
        raise ValueError("At least one positive maximum depth is required.")
    if not leaf_values or any(value < 1 for value in leaf_values):
        raise ValueError("At least one positive minimum leaf size is required.")
    if not feature_values or any(not 0 < value <= 1 for value in feature_values):
        raise ValueError("At least one maximum feature fraction in (0, 1] is required.")

    features = training.loc[:, MODEL_FEATURES]
    target = training[TARGET_COLUMN]
    splits = make_expanding_window_splits(training[YEAR_COLUMN])
    rows: list[dict[str, float | int | str]] = []

    for max_depth in depth_values:
        for min_samples_leaf in leaf_values:
            for max_features in feature_values:
                for fold, (train_indices, validation_indices) in zip(
                    EXPANDING_FOLDS,
                    splits,
                    strict=True,
                ):
                    pipeline = make_random_forest_pipeline(
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        n_estimators=n_estimators,
                    )
                    pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
                    probabilities = pipeline.predict_proba(
                        features.iloc[validation_indices]
                    )[:, 1]
                    validation_target = target.iloc[validation_indices]
                    forest = pipeline.named_steps["classifier"]
                    mean_depth = np.mean(
                        [estimator.tree_.max_depth for estimator in forest.estimators_]
                    )
                    mean_leaves = np.mean(
                        [estimator.tree_.n_leaves for estimator in forest.estimators_]
                    )

                    rows.append(
                        {
                            "model": MODEL_NAME,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                            "n_estimators": n_estimators,
                            "fold": fold.name,
                            "train_end_year": fold.train_end,
                            "validation_start_year": fold.validation_start,
                            "validation_end_year": fold.validation_end,
                            "train_observations": len(train_indices),
                            "validation_observations": len(validation_indices),
                            "validation_bankruptcies": int(validation_target.sum()),
                            "mean_fitted_tree_depth": mean_depth,
                            "mean_fitted_leaf_count": mean_leaves,
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


def select_random_forest(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Choose a competitive forest using the one-standard-error rule.

    Candidates within one standard error of the best mean PR-AUC are considered
    equivalent. The candidate with fewer average leaves is preferred, followed by
    shallower trees, larger minimum leaves, and fewer candidate features per split.
    """
    required_columns = {
        "model",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "n_estimators",
        "mean_fitted_tree_depth",
        "mean_fitted_leaf_count",
        "pr_auc",
        "roc_auc",
    }
    missing_columns = required_columns.difference(cv_results.columns)
    if missing_columns:
        raise ValueError(f"Forest cross-validation columns are missing: {missing_columns}")

    summary = (
        cv_results.groupby(
            [
                "model",
                "max_depth",
                "min_samples_leaf",
                "max_features",
                "n_estimators",
            ],
            as_index=False,
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_fitted_depth=("mean_fitted_tree_depth", "mean"),
            mean_fitted_leaf_count=("mean_fitted_leaf_count", "mean"),
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
            "min_samples_leaf",
            "max_features",
            "max_depth",
        ],
        ascending=[True, True, False, True, True],
    ).head(1)
    selected["best_mean_pr_auc"] = best["mean_pr_auc"]
    selected["one_standard_error"] = standard_error
    selected["eligibility_threshold"] = eligibility_threshold
    selected["selection_rule"] = "one_standard_error_simplest_forest"
    return selected.reset_index(drop=True)


def fit_selected_random_forest(
    model_data: pd.DataFrame,
    selection: pd.DataFrame,
    n_estimators: int = FINAL_ESTIMATORS,
) -> Pipeline:
    """Fit the selected forest on all training years with out-of-bag predictions."""
    _validate_model_data(model_data)
    if len(selection) != 1:
        raise ValueError("Random Forest selection must contain exactly one candidate.")
    training = model_data.loc[model_data["sample_period"].eq("train")]
    if training.empty:
        raise ValueError("Training observations are required to fit the final forest.")

    selected = selection.iloc[0]
    pipeline = make_random_forest_pipeline(
        max_depth=int(selected["max_depth"]),
        min_samples_leaf=int(selected["min_samples_leaf"]),
        max_features=float(selected["max_features"]),
        n_estimators=n_estimators,
        oob_score=True,
    )
    pipeline.fit(training.loc[:, MODEL_FEATURES], training[TARGET_COLUMN])
    return pipeline


def build_random_forest_oob_diagnostics(
    model_data: pd.DataFrame,
    fitted_forest: Pipeline,
) -> pd.DataFrame:
    """Summarize out-of-bag discrimination and probability accuracy."""
    _validate_model_data(model_data)
    training = model_data.loc[model_data["sample_period"].eq("train")]
    target = training[TARGET_COLUMN].to_numpy()
    classifier = fitted_forest.named_steps["classifier"]
    if not hasattr(classifier, "oob_decision_function_"):
        raise ValueError("The fitted forest does not contain out-of-bag predictions.")
    probabilities = classifier.oob_decision_function_[:, 1]
    if len(probabilities) != len(target) or not np.isfinite(probabilities).all():
        raise ValueError("Out-of-bag probabilities are incomplete or non-finite.")

    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "observations": len(target),
                "bankruptcies": int(target.sum()),
                "event_rate": float(target.mean()),
                "oob_pr_auc": average_precision_score(target, probabilities),
                "oob_roc_auc": roc_auc_score(target, probabilities),
                "oob_brier_score": brier_score_loss(target, probabilities),
            }
        ]
    )


def build_random_forest_feature_importance(fitted_forest: Pipeline) -> pd.DataFrame:
    """Return mean impurity importance and its variability across trees."""
    classifier = fitted_forest.named_steps["classifier"]
    tree_importances = np.vstack(
        [estimator.feature_importances_ for estimator in classifier.estimators_]
    )
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": classifier.feature_importances_,
            "importance_std_across_trees": tree_importances.std(axis=0),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance.loc[
        :, ["rank", "feature", "importance", "importance_std_across_trees"]
    ]


def create_random_forest_validation_predictions(
    model_data: pd.DataFrame,
    fitted_forest: Pipeline,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create Random Forest probabilities for the 2012-2014 validation period."""
    _validate_model_data(model_data)
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must lie between zero and one.")
    validation = model_data.loc[model_data["sample_period"].eq("validation")]
    if validation.empty:
        raise ValueError("Validation observations are required to create predictions.")

    probabilities = fitted_forest.predict_proba(validation.loc[:, MODEL_FEATURES])[:, 1]
    predictions = validation.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN]].copy()
    predictions["model"] = MODEL_NAME
    predictions["predicted_probability"] = probabilities
    predictions["predicted_class"] = (probabilities >= threshold).astype("int8")
    return predictions.reset_index(drop=True)


def plot_random_forest_feature_importance(
    importance: pd.DataFrame,
    top_n: int = 15,
) -> Figure:
    """Plot the most important Random Forest predictors with tree-level variation."""
    if top_n < 1:
        raise ValueError("At least one feature must be displayed.")
    displayed = importance.head(top_n).sort_values("importance", ascending=True)
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.barh(
        displayed["feature"].str.replace("_", " ").str.title(),
        displayed["importance"],
        xerr=displayed["importance_std_across_trees"],
        color="#1F4E79",
        alpha=0.85,
        error_kw={"ecolor": "#666666", "capsize": 2, "elinewidth": 0.8},
    )
    axis.set_title("Random Forest Feature Importance")
    axis.set_xlabel("Mean decrease in impurity")
    axis.set_ylabel("")
    figure.tight_layout()
    return figure
