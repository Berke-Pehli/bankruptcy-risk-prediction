from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankruptcy_risk.features import INTERPRETABLE_FEATURES
from bankruptcy_risk.interpretable_logit import (
    build_coefficient_table,
    build_marginal_effects_table,
    create_validation_logit_predictions,
    fit_interpretable_logit,
)
from bankruptcy_risk.target import TARGET_COLUMN, YEAR_COLUMN


@pytest.fixture(scope="module")
def synthetic_model_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    observations = 800
    data = pd.DataFrame(
        rng.normal(size=(observations, len(INTERPRETABLE_FEATURES))),
        columns=INTERPRETABLE_FEATURES,
    )
    linear_predictor = -2.0 + 0.7 * data["debt_to_assets"] - 0.5 * data["return_on_assets"]
    event_probability = 1 / (1 + np.exp(-linear_predictor))
    data[TARGET_COLUMN] = rng.binomial(1, event_probability)
    data["company_id"] = [f"company_{index // 4}" for index in range(observations)]
    data[YEAR_COLUMN] = np.where(np.arange(observations) < 600, 2011, 2012)
    data["sample_period"] = np.where(np.arange(observations) < 600, "train", "validation")
    return data


@pytest.fixture(scope="module")
def fitted_logit(synthetic_model_data: pd.DataFrame):
    return fit_interpretable_logit(synthetic_model_data)


def test_validation_values_cannot_change_fitted_parameters(
    synthetic_model_data: pd.DataFrame,
    fitted_logit,
) -> None:
    modified = synthetic_model_data.copy()
    validation = modified["sample_period"].eq("validation")
    modified.loc[validation, INTERPRETABLE_FEATURES] = 1_000_000

    refitted = fit_interpretable_logit(modified)

    np.testing.assert_allclose(refitted.result.params, fitted_logit.result.params)


def test_validation_predictions_are_probabilities(
    synthetic_model_data: pd.DataFrame,
    fitted_logit,
) -> None:
    predictions = create_validation_logit_predictions(synthetic_model_data, fitted_logit)

    assert len(predictions) == 200
    assert predictions["model"].eq("interpretable_logit").all()
    assert predictions["predicted_probability"].between(0, 1).all()
    assert set(predictions["predicted_class"]).issubset({0, 1})


def test_coefficient_table_includes_intercept_and_odds_ratios(fitted_logit) -> None:
    coefficients = build_coefficient_table(fitted_logit)

    assert coefficients["feature"].tolist() == ["const", *INTERPRETABLE_FEATURES]
    assert coefficients["odds_ratio"].gt(0).all()
    assert coefficients["standard_error"].gt(0).all()


def test_marginal_effects_cover_each_financial_predictor(fitted_logit) -> None:
    marginal_effects = build_marginal_effects_table(fitted_logit)

    assert marginal_effects["feature"].tolist() == list(INTERPRETABLE_FEATURES)
    assert marginal_effects["average_marginal_effect"].notna().all()
