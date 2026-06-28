from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankruptcy_risk.features import MODEL_FEATURES
from bankruptcy_risk.regularized_logit import (
    MODEL_PENALTIES,
    build_regularized_coefficient_table,
    create_regularized_validation_predictions,
    cross_validate_regularized_logits,
    fit_selected_regularized_logits,
    select_regularization_strengths,
)
from bankruptcy_risk.target import TARGET_COLUMN, YEAR_COLUMN


@pytest.fixture(scope="module")
def synthetic_model_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    observations_per_year = 60
    years = np.repeat(np.arange(1999, 2015), observations_per_year)
    data = pd.DataFrame(
        rng.normal(size=(len(years), len(MODEL_FEATURES))),
        columns=MODEL_FEATURES,
    )
    linear_predictor = (
        -2.0
        + 0.8 * data["debt_to_assets"]
        - 0.6 * data["return_on_assets"]
        - 0.4 * data["signed_log_total_assets"]
    )
    probabilities = 1 / (1 + np.exp(-linear_predictor))
    data[TARGET_COLUMN] = rng.binomial(1, probabilities)
    data["company_id"] = [f"company_{index // 4}" for index in range(len(data))]
    data[YEAR_COLUMN] = years
    data["sample_period"] = np.where(years <= 2011, "train", "validation")
    return data


@pytest.fixture(scope="module")
def model_selection(synthetic_model_data: pd.DataFrame) -> pd.DataFrame:
    cv_results = cross_validate_regularized_logits(
        synthetic_model_data,
        c_values=(0.1, 1.0),
    )
    return select_regularization_strengths(cv_results)


@pytest.fixture(scope="module")
def fitted_models(
    synthetic_model_data: pd.DataFrame,
    model_selection: pd.DataFrame,
):
    return fit_selected_regularized_logits(synthetic_model_data, model_selection)


def test_cross_validation_covers_models_candidates_and_folds(
    synthetic_model_data: pd.DataFrame,
) -> None:
    results = cross_validate_regularized_logits(
        synthetic_model_data,
        c_values=(0.1, 1.0),
    )

    assert len(results) == 2 * 2 * 4
    assert set(results["model"]) == set(MODEL_PENALTIES)
    assert results["train_end_year"].lt(results["validation_start_year"]).all()
    assert results["pr_auc"].between(0, 1).all()
    assert results["roc_auc"].between(0, 1).all()


def test_selection_returns_one_candidate_per_model(model_selection: pd.DataFrame) -> None:
    assert model_selection["model"].is_unique
    assert set(model_selection["model"]) == set(MODEL_PENALTIES)
    assert model_selection["folds"].eq(4).all()


def test_duplicate_model_selection_is_rejected(
    synthetic_model_data: pd.DataFrame,
    model_selection: pd.DataFrame,
) -> None:
    duplicate_selection = pd.concat(
        [model_selection, model_selection.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="exactly one Ridge and one Lasso"):
        fit_selected_regularized_logits(synthetic_model_data, duplicate_selection)


def test_validation_values_cannot_change_final_model_coefficients(
    synthetic_model_data: pd.DataFrame,
    model_selection: pd.DataFrame,
    fitted_models,
) -> None:
    modified = synthetic_model_data.copy()
    validation = modified["sample_period"].eq("validation")
    modified.loc[validation, MODEL_FEATURES] = 1_000_000
    refitted_models = fit_selected_regularized_logits(modified, model_selection)

    for model_name in MODEL_PENALTIES:
        original = fitted_models[model_name].named_steps["classifier"].coef_
        refitted = refitted_models[model_name].named_steps["classifier"].coef_
        np.testing.assert_allclose(original, refitted)


def test_coefficient_table_covers_all_features_and_models(fitted_models) -> None:
    coefficients = build_regularized_coefficient_table(fitted_models)

    assert len(coefficients) == 2 * len(MODEL_FEATURES)
    assert set(coefficients["feature"]) == set(MODEL_FEATURES)
    assert coefficients["odds_ratio_per_standard_deviation"].gt(0).all()


def test_validation_predictions_cover_both_models(
    synthetic_model_data: pd.DataFrame,
    fitted_models,
) -> None:
    predictions = create_regularized_validation_predictions(
        synthetic_model_data,
        fitted_models,
    )

    expected_validation_rows = synthetic_model_data["sample_period"].eq("validation").sum()
    assert len(predictions) == 2 * expected_validation_rows
    assert set(predictions["model"]) == set(MODEL_PENALTIES)
    assert predictions["predicted_probability"].between(0, 1).all()
