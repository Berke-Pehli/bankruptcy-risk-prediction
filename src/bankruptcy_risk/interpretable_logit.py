"""Fit and explain the interpretable Logistic Regression benchmark.

This module estimates one-year-ahead bankruptcy probabilities from a compact set of
financial indicators. The specification represents firm size, liquidity, leverage,
profitability, accumulated earnings, operating efficiency, and market valuation.

Input:
    - Training observations from ``data/processed/model_dataset.parquet``.

Outputs:
    - A fitted preprocessing pipeline and Logistic Regression result.
    - Standardized coefficients, odds ratios, and company-clustered confidence intervals.
    - Average marginal effects expressed as changes in predicted bankruptcy probability.
    - Predicted probabilities for the 2012-2014 validation period.

Preprocessing is estimated exclusively on 1999-2011 observations. Standard errors are
clustered by company because a company can contribute several annual observations. The
reported associations support financial interpretation but should not be read as causal
effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.pipeline import Pipeline

from bankruptcy_risk.features import INTERPRETABLE_FEATURES
from bankruptcy_risk.preprocessing import make_linear_preprocessor
from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN


@dataclass
class InterpretableLogitFit:
    """Store the fitted preprocessing pipeline and statistical model result."""

    preprocessor: Pipeline
    result: Any
    feature_names: tuple[str, ...]


def _validate_model_data(model_data: pd.DataFrame) -> None:
    """Check that the temporal model table contains all benchmark inputs."""
    required_columns = {
        COMPANY_COLUMN,
        YEAR_COLUMN,
        TARGET_COLUMN,
        "sample_period",
        *INTERPRETABLE_FEATURES,
    }
    missing_columns = required_columns.difference(model_data.columns)
    if missing_columns:
        raise ValueError(
            f"Required interpretable-logit columns are missing: {sorted(missing_columns)}"
        )


def fit_interpretable_logit(model_data: pd.DataFrame) -> InterpretableLogitFit:
    """Estimate the benchmark Logistic Regression on training observations only.

    Financial predictors are median-imputed, clipped, and standardized using statistics
    from the training period. Coefficients therefore describe a one-training-standard-
    deviation increase in each predictor, holding the remaining predictors constant.
    Company-clustered standard errors account for repeated firm observations.
    """
    _validate_model_data(model_data)
    training = model_data.loc[model_data["sample_period"].eq("train")].copy()
    if training.empty:
        raise ValueError("Training observations are required to fit the benchmark model.")
    if training[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The training target must contain both bankruptcy classes.")

    preprocessor = make_linear_preprocessor()
    transformed = preprocessor.fit_transform(training.loc[:, INTERPRETABLE_FEATURES])
    design = pd.DataFrame(
        transformed,
        columns=INTERPRETABLE_FEATURES,
        index=training.index,
    )
    design = sm.add_constant(design, has_constant="add")

    result = sm.Logit(training[TARGET_COLUMN].astype(float), design).fit(
        disp=False,
        maxiter=200,
        cov_type="cluster",
        cov_kwds={"groups": training[COMPANY_COLUMN]},
    )
    if not result.mle_retvals.get("converged", False):
        raise RuntimeError("The interpretable Logistic Regression did not converge.")

    return InterpretableLogitFit(
        preprocessor=preprocessor,
        result=result,
        feature_names=INTERPRETABLE_FEATURES,
    )


def build_coefficient_table(fitted: InterpretableLogitFit) -> pd.DataFrame:
    """Return coefficients, odds ratios, and clustered uncertainty estimates."""
    result = fitted.result
    confidence_intervals = result.conf_int()
    table = pd.DataFrame(
        {
            "feature": result.params.index,
            "coefficient": result.params.to_numpy(),
            "standard_error": result.bse.to_numpy(),
            "z_value": result.tvalues.to_numpy(),
            "p_value": result.pvalues.to_numpy(),
            "confidence_interval_lower": confidence_intervals.iloc[:, 0].to_numpy(),
            "confidence_interval_upper": confidence_intervals.iloc[:, 1].to_numpy(),
        }
    )
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["odds_ratio_ci_lower"] = np.exp(table["confidence_interval_lower"])
    table["odds_ratio_ci_upper"] = np.exp(table["confidence_interval_upper"])
    return table


def build_marginal_effects_table(fitted: InterpretableLogitFit) -> pd.DataFrame:
    """Return average marginal effects with company-clustered uncertainty."""
    effects = fitted.result.get_margeff(at="overall", method="dydx")
    confidence_intervals = effects.conf_int()
    return pd.DataFrame(
        {
            "feature": fitted.feature_names,
            "average_marginal_effect": effects.margeff,
            "standard_error": effects.margeff_se,
            "z_value": effects.tvalues,
            "p_value": effects.pvalues,
            "confidence_interval_lower": confidence_intervals[:, 0],
            "confidence_interval_upper": confidence_intervals[:, 1],
        }
    )


def create_validation_logit_predictions(
    model_data: pd.DataFrame,
    fitted: InterpretableLogitFit,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict bankruptcy probabilities for the validation period.

    The conventional 0.50 classification threshold is retained only as an initial
    reference. A later evaluation milestone will select decision thresholds explicitly.
    """
    _validate_model_data(model_data)
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must lie between zero and one.")

    validation = model_data.loc[model_data["sample_period"].eq("validation")].copy()
    if validation.empty:
        raise ValueError("Validation observations are required to create predictions.")

    transformed = fitted.preprocessor.transform(validation.loc[:, fitted.feature_names])
    design = pd.DataFrame(
        transformed,
        columns=fitted.feature_names,
        index=validation.index,
    )
    design = sm.add_constant(design, has_constant="add")
    probabilities = np.asarray(fitted.result.predict(design), dtype=float)

    predictions = validation.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN]].copy()
    predictions["model"] = "interpretable_logit"
    predictions["predicted_probability"] = probabilities
    predictions["predicted_class"] = (probabilities >= threshold).astype("int8")
    return predictions.reset_index(drop=True)
