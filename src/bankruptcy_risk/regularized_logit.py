"""Select and fit Ridge- and Lasso-regularized Logistic Regression models.

This module extends the compact statistical benchmark to all engineered predictors.
Regularization controls model complexity when accounting measures and financial ratios
are correlated.

Inputs:
    - Training observations from ``data/processed/model_dataset.parquet``.
    - The four expanding-window folds defined in ``splitting.py``.

Models:
    - Ridge Logistic Regression applies an L2 penalty that shrinks coefficients.
    - Lasso Logistic Regression applies an L1 penalty that can set coefficients to zero.

Model selection:
    - Candidate penalty strengths are compared using expanding-window validation.
    - PR-AUC is the primary score because bankruptcies are rare.
    - ROC-AUC is retained as a complementary ranking measure.
    - Every fold fits imputation, clipping, and scaling on its own training years.

Outputs:
    - Fold-level cross-validation results and selected penalty strengths.
    - Final models fitted on 1999-2011 observations only.
    - Standardized coefficients and 2012-2014 validation probabilities.

The models are intentionally unweighted. This preserves the observed event prevalence
in their probability estimates. Classification thresholds and probability calibration
are handled explicitly in later evaluation stages.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from bankruptcy_risk.config import RANDOM_SEED
from bankruptcy_risk.features import MODEL_FEATURES
from bankruptcy_risk.preprocessing import make_linear_preprocessor
from bankruptcy_risk.splitting import EXPANDING_FOLDS, make_expanding_window_splits
from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN

DEFAULT_C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
MODEL_PENALTIES = {
    "ridge_logit": "l2",
    "lasso_logit": "l1",
}


def _validate_model_data(model_data: pd.DataFrame) -> None:
    """Check that the temporal model table contains every required input."""
    required_columns = {
        COMPANY_COLUMN,
        YEAR_COLUMN,
        TARGET_COLUMN,
        "sample_period",
        *MODEL_FEATURES,
    }
    missing_columns = required_columns.difference(model_data.columns)
    if missing_columns:
        raise ValueError(
            f"Required regularized-logit columns are missing: {sorted(missing_columns)}"
        )


def make_regularized_logit_pipeline(
    penalty: str,
    c_value: float,
) -> Pipeline:
    """Create a leakage-safe regularized Logistic Regression pipeline.

    Parameters
    ----------
    penalty:
        ``"l1"`` for Lasso or ``"l2"`` for Ridge regularization.
    c_value:
        Inverse regularization strength. Smaller values imply stronger shrinkage.

    """
    if penalty not in {"l1", "l2"}:
        raise ValueError("Penalty must be either 'l1' or 'l2'.")
    if c_value <= 0:
        raise ValueError("Inverse regularization strength C must be positive.")

    return Pipeline(
        steps=[
            ("preprocessor", make_linear_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    l1_ratio=1.0 if penalty == "l1" else 0.0,
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def cross_validate_regularized_logits(
    model_data: pd.DataFrame,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
) -> pd.DataFrame:
    """Evaluate Ridge and Lasso candidates with expanding-window validation.

    Only observations marked as training data are considered. Each candidate pipeline
    is fitted independently inside each fold so no imputation, clipping, scaling, or
    coefficient estimate can use the fold's future observations.
    """
    _validate_model_data(model_data)
    training = model_data.loc[model_data["sample_period"].eq("train")].reset_index(
        drop=True
    )
    if training.empty:
        raise ValueError("Training observations are required for model selection.")

    candidate_values = tuple(float(value) for value in c_values)
    if not candidate_values or any(value <= 0 for value in candidate_values):
        raise ValueError("At least one positive candidate C value is required.")

    features = training.loc[:, MODEL_FEATURES]
    target = training[TARGET_COLUMN]
    splits = make_expanding_window_splits(training[YEAR_COLUMN])
    rows: list[dict[str, float | int | str]] = []

    for model_name, penalty in MODEL_PENALTIES.items():
        for c_value in candidate_values:
            for fold, (train_indices, validation_indices) in zip(
                EXPANDING_FOLDS,
                splits,
                strict=True,
            ):
                pipeline = make_regularized_logit_pipeline(penalty, c_value)
                pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
                probabilities = pipeline.predict_proba(
                    features.iloc[validation_indices]
                )[:, 1]
                validation_target = target.iloc[validation_indices]

                rows.append(
                    {
                        "model": model_name,
                        "penalty": penalty,
                        "inverse_penalty_strength_c": c_value,
                        "penalty_strength_lambda": 1 / c_value,
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


def select_regularization_strengths(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Select one penalty strength per model using mean expanding-fold PR-AUC."""
    required_columns = {
        "model",
        "penalty",
        "inverse_penalty_strength_c",
        "penalty_strength_lambda",
        "pr_auc",
        "roc_auc",
    }
    missing_columns = required_columns.difference(cv_results.columns)
    if missing_columns:
        raise ValueError(f"Cross-validation columns are missing: {sorted(missing_columns)}")

    summary = (
        cv_results.groupby(
            [
                "model",
                "penalty",
                "inverse_penalty_strength_c",
                "penalty_strength_lambda",
            ],
            as_index=False,
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            folds=("pr_auc", "size"),
        )
        .sort_values(
            ["model", "mean_pr_auc", "std_pr_auc", "inverse_penalty_strength_c"],
            ascending=[True, False, True, True],
        )
    )
    selected = summary.groupby("model", as_index=False, sort=False).head(1)
    return selected.sort_values("model", ignore_index=True)


def fit_selected_regularized_logits(
    model_data: pd.DataFrame,
    selection: pd.DataFrame,
) -> dict[str, Pipeline]:
    """Fit selected Ridge and Lasso pipelines on the complete training period."""
    _validate_model_data(model_data)
    training = model_data.loc[model_data["sample_period"].eq("train")]
    if training.empty:
        raise ValueError("Training observations are required to fit final models.")

    selected_models = set(selection.get("model", pd.Series(dtype=str)))
    if (
        len(selection) != len(MODEL_PENALTIES)
        or not selection["model"].is_unique
        or selected_models != set(MODEL_PENALTIES)
    ):
        raise ValueError("Selection must contain exactly one Ridge and one Lasso model.")

    fitted_models: dict[str, Pipeline] = {}
    for row in selection.itertuples(index=False):
        expected_penalty = MODEL_PENALTIES[row.model]
        if row.penalty != expected_penalty:
            raise ValueError(f"Selected penalty does not match {row.model}.")
        pipeline = make_regularized_logit_pipeline(
            penalty=row.penalty,
            c_value=row.inverse_penalty_strength_c,
        )
        pipeline.fit(
            training.loc[:, MODEL_FEATURES],
            training[TARGET_COLUMN],
        )
        fitted_models[row.model] = pipeline
    return fitted_models


def build_regularized_coefficient_table(
    fitted_models: dict[str, Pipeline],
) -> pd.DataFrame:
    """Return standardized coefficients and Lasso feature-selection indicators."""
    rows: list[dict[str, float | bool | str]] = []
    for model_name, pipeline in fitted_models.items():
        classifier = pipeline.named_steps["classifier"]
        coefficients = classifier.coef_[0]
        for feature, coefficient in zip(MODEL_FEATURES, coefficients, strict=True):
            rows.append(
                {
                    "model": model_name,
                    "penalty": MODEL_PENALTIES[model_name],
                    "inverse_penalty_strength_c": classifier.C,
                    "intercept": classifier.intercept_[0],
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                    "odds_ratio_per_standard_deviation": np.exp(coefficient),
                    "is_nonzero": not np.isclose(coefficient, 0.0, atol=1e-12),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model", "absolute_coefficient"],
        ascending=[True, False],
        ignore_index=True,
    )


def create_regularized_validation_predictions(
    model_data: pd.DataFrame,
    fitted_models: dict[str, Pipeline],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create Ridge and Lasso bankruptcy probabilities for validation years."""
    _validate_model_data(model_data)
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must lie between zero and one.")

    validation = model_data.loc[model_data["sample_period"].eq("validation")]
    if validation.empty:
        raise ValueError("Validation observations are required to create predictions.")

    identifiers = validation.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN]]
    predictions: list[pd.DataFrame] = []
    for model_name, pipeline in fitted_models.items():
        probabilities = pipeline.predict_proba(validation.loc[:, MODEL_FEATURES])[:, 1]
        model_predictions = identifiers.copy()
        model_predictions["model"] = model_name
        model_predictions["predicted_probability"] = probabilities
        model_predictions["predicted_class"] = (probabilities >= threshold).astype("int8")
        predictions.append(model_predictions)
    return pd.concat(predictions, ignore_index=True)
