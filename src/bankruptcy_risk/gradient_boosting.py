"""Select, fit, and explain a Gradient Boosting bankruptcy classifier.

Gradient Boosting builds an additive classifier sequentially. Each shallow tree focuses
on errors left by the preceding ensemble, allowing nonlinear effects and interactions to
emerge gradually rather than through one large decision tree.

Inputs:
    - Training observations from ``data/processed/model_dataset.parquet``.
    - The expanding-window folds defined in ``splitting.py``.

Model selection:
    - Learning rate controls the contribution of each weak tree.
    - The number of estimators controls how many boosting stages are added.
    - Tree depth determines whether components represent main effects or interactions.
    - PR-AUC is the primary score and ROC-AUC is recorded for comparison.
    - The one-standard-error rule favors a simpler competitive ensemble.

Outputs:
    - Fold-level tuning results and selected hyperparameters.
    - A final model fitted on 1999-2011 observations only.
    - Validation probabilities and ranked impurity-based feature importances.

Every candidate learns median imputation and clipping within its own training fold.
Component trees are shallow, each fit uses an 80% stochastic subsample, and only half of
the predictors are considered at a split. These controls reduce overfitting and runtime.
The model is unweighted so probability calibration can be assessed transparently later.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from bankruptcy_risk.config import RANDOM_SEED
from bankruptcy_risk.features import MODEL_FEATURES
from bankruptcy_risk.preprocessing import make_tree_preprocessor
from bankruptcy_risk.splitting import EXPANDING_FOLDS, make_expanding_window_splits
from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN

DEFAULT_LEARNING_RATES = (0.03, 0.1)
DEFAULT_N_ESTIMATORS = (100, 200)
DEFAULT_MAX_DEPTHS = (1, 2)
MIN_SAMPLES_LEAF = 50
SUBSAMPLE = 0.8
MAX_FEATURES = 0.5
MODEL_NAME = "gradient_boosting"


def _validate_model_data(model_data: pd.DataFrame) -> None:
    """Check that the temporal model table contains every boosting input."""
    required_columns = {
        COMPANY_COLUMN,
        YEAR_COLUMN,
        TARGET_COLUMN,
        "sample_period",
        *MODEL_FEATURES,
    }
    missing_columns = required_columns.difference(model_data.columns)
    if missing_columns:
        raise ValueError(f"Required Gradient Boosting columns are missing: {missing_columns}")


def make_gradient_boosting_pipeline(
    learning_rate: float,
    n_estimators: int,
    max_depth: int,
) -> Pipeline:
    """Create a leakage-safe preprocessing and Gradient Boosting pipeline."""
    if learning_rate <= 0:
        raise ValueError("The learning rate must be positive.")
    if n_estimators < 1:
        raise ValueError("At least one boosting stage is required.")
    if max_depth < 1:
        raise ValueError("Weak-tree depth must be positive.")

    return Pipeline(
        steps=[
            ("preprocessor", make_tree_preprocessor()),
            (
                "classifier",
                GradientBoostingClassifier(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    subsample=SUBSAMPLE,
                    max_features=MAX_FEATURES,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def cross_validate_gradient_boosting(
    model_data: pd.DataFrame,
    learning_rates: Iterable[float] = DEFAULT_LEARNING_RATES,
    n_estimators_values: Iterable[int] = DEFAULT_N_ESTIMATORS,
    max_depths: Iterable[int] = DEFAULT_MAX_DEPTHS,
) -> pd.DataFrame:
    """Evaluate boosting candidates with expanding-window cross-validation."""
    _validate_model_data(model_data)
    training = model_data.loc[model_data["sample_period"].eq("train")].reset_index(
        drop=True
    )
    if training.empty:
        raise ValueError("Training observations are required for boosting selection.")

    rate_values = tuple(float(value) for value in learning_rates)
    estimator_values = tuple(int(value) for value in n_estimators_values)
    depth_values = tuple(int(value) for value in max_depths)
    if not rate_values or any(value <= 0 for value in rate_values):
        raise ValueError("At least one positive learning rate is required.")
    if not estimator_values or any(value < 1 for value in estimator_values):
        raise ValueError("At least one positive estimator count is required.")
    if not depth_values or any(value < 1 for value in depth_values):
        raise ValueError("At least one positive weak-tree depth is required.")

    features = training.loc[:, MODEL_FEATURES]
    target = training[TARGET_COLUMN]
    splits = make_expanding_window_splits(training[YEAR_COLUMN])
    rows: list[dict[str, float | int | str]] = []

    for learning_rate in rate_values:
        for n_estimators in estimator_values:
            for max_depth in depth_values:
                for fold, (train_indices, validation_indices) in zip(
                    EXPANDING_FOLDS,
                    splits,
                    strict=True,
                ):
                    pipeline = make_gradient_boosting_pipeline(
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                    )
                    pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
                    probabilities = pipeline.predict_proba(
                        features.iloc[validation_indices]
                    )[:, 1]
                    validation_target = target.iloc[validation_indices]

                    rows.append(
                        {
                            "model": MODEL_NAME,
                            "learning_rate": learning_rate,
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_leaf": MIN_SAMPLES_LEAF,
                            "subsample": SUBSAMPLE,
                            "max_features": MAX_FEATURES,
                            "fold": fold.name,
                            "train_end_year": fold.train_end,
                            "validation_start_year": fold.validation_start,
                            "validation_end_year": fold.validation_end,
                            "train_observations": len(train_indices),
                            "validation_observations": len(validation_indices),
                            "validation_bankruptcies": int(validation_target.sum()),
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


def select_gradient_boosting(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Choose a competitive ensemble using the one-standard-error rule.

    Candidates within one standard error of the best mean PR-AUC are eligible. The
    smallest approximate ensemble size, measured by estimators times potential leaves,
    is preferred, followed by fewer stages, shallower trees, and a lower learning rate.
    """
    required_columns = {
        "model",
        "learning_rate",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "max_features",
        "pr_auc",
        "roc_auc",
    }
    missing_columns = required_columns.difference(cv_results.columns)
    if missing_columns:
        raise ValueError(
            f"Boosting cross-validation columns are missing: {missing_columns}"
        )

    parameter_columns = [
        "model",
        "learning_rate",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "max_features",
    ]
    summary = (
        cv_results.groupby(parameter_columns, as_index=False)
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            folds=("pr_auc", "size"),
        )
    )
    summary["approximate_terminal_regions"] = summary["n_estimators"] * (
        2 ** summary["max_depth"]
    )
    best = summary.loc[summary["mean_pr_auc"].idxmax()]
    standard_error = best["std_pr_auc"] / np.sqrt(best["folds"])
    eligibility_threshold = best["mean_pr_auc"] - standard_error
    eligible = summary.loc[summary["mean_pr_auc"].ge(eligibility_threshold)].copy()
    selected = eligible.sort_values(
        [
            "approximate_terminal_regions",
            "n_estimators",
            "max_depth",
            "learning_rate",
        ],
        ascending=[True, True, True, True],
    ).head(1)
    selected["best_mean_pr_auc"] = best["mean_pr_auc"]
    selected["one_standard_error"] = standard_error
    selected["eligibility_threshold"] = eligibility_threshold
    selected["selection_rule"] = "one_standard_error_simplest_boosting"
    return selected.reset_index(drop=True)


def fit_selected_gradient_boosting(
    model_data: pd.DataFrame,
    selection: pd.DataFrame,
) -> Pipeline:
    """Fit the selected boosting pipeline on all 1999-2011 observations."""
    _validate_model_data(model_data)
    if len(selection) != 1:
        raise ValueError("Gradient Boosting selection must contain exactly one candidate.")
    training = model_data.loc[model_data["sample_period"].eq("train")]
    if training.empty:
        raise ValueError("Training observations are required to fit final boosting.")

    selected = selection.iloc[0]
    pipeline = make_gradient_boosting_pipeline(
        learning_rate=float(selected["learning_rate"]),
        n_estimators=int(selected["n_estimators"]),
        max_depth=int(selected["max_depth"]),
    )
    pipeline.fit(training.loc[:, MODEL_FEATURES], training[TARGET_COLUMN])
    return pipeline


def build_gradient_boosting_feature_importance(
    fitted_boosting: Pipeline,
) -> pd.DataFrame:
    """Return ranked impurity-based importance across the boosted weak trees."""
    classifier = fitted_boosting.named_steps["classifier"]
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance["used_by_ensemble"] = importance["importance"].gt(0)
    return importance.loc[:, ["rank", "feature", "importance", "used_by_ensemble"]]


def create_gradient_boosting_validation_predictions(
    model_data: pd.DataFrame,
    fitted_boosting: Pipeline,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create Gradient Boosting probabilities for the validation period."""
    _validate_model_data(model_data)
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must lie between zero and one.")
    validation = model_data.loc[model_data["sample_period"].eq("validation")]
    if validation.empty:
        raise ValueError("Validation observations are required to create predictions.")

    probabilities = fitted_boosting.predict_proba(validation.loc[:, MODEL_FEATURES])[:, 1]
    predictions = validation.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN]].copy()
    predictions["model"] = MODEL_NAME
    predictions["predicted_probability"] = probabilities
    predictions["predicted_class"] = (probabilities >= threshold).astype("int8")
    return predictions.reset_index(drop=True)


def plot_gradient_boosting_feature_importance(
    importance: pd.DataFrame,
    top_n: int = 15,
) -> Figure:
    """Plot the most important predictors in the fitted boosting ensemble."""
    if top_n < 1:
        raise ValueError("At least one feature must be displayed.")
    displayed = importance.head(top_n).sort_values("importance", ascending=True)
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.barh(
        displayed["feature"].str.replace("_", " ").str.title(),
        displayed["importance"],
        color="#C55A11",
        alpha=0.85,
    )
    axis.set_title("Gradient Boosting Feature Importance")
    axis.set_xlabel("Relative mean decrease in impurity")
    axis.set_ylabel("")
    figure.tight_layout()
    return figure
