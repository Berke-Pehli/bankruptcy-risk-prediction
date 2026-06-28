from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bankruptcy_risk.features import MODEL_FEATURES
from bankruptcy_risk.gradient_boosting import (
    MODEL_NAME,
    build_gradient_boosting_feature_importance,
    create_gradient_boosting_validation_predictions,
    cross_validate_gradient_boosting,
    fit_selected_gradient_boosting,
    plot_gradient_boosting_feature_importance,
    select_gradient_boosting,
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
    event_signal = (
        (data["debt_to_assets"] > 0.6)
        & (data["return_on_assets"] < -0.2)
    ) | (data["signed_log_total_assets"] < -1.4)
    data[TARGET_COLUMN] = np.where(event_signal, 1, rng.binomial(1, 0.03, len(data)))
    data["company_id"] = [f"company_{index // 4}" for index in range(len(data))]
    data[YEAR_COLUMN] = years
    data["sample_period"] = np.where(years <= 2011, "train", "validation")
    return data


@pytest.fixture(scope="module")
def boosting_selection(synthetic_model_data: pd.DataFrame) -> pd.DataFrame:
    cv_results = cross_validate_gradient_boosting(
        synthetic_model_data,
        learning_rates=(0.05, 0.1),
        n_estimators_values=(20,),
        max_depths=(1, 2),
    )
    return select_gradient_boosting(cv_results)


@pytest.fixture(scope="module")
def fitted_boosting(
    synthetic_model_data: pd.DataFrame,
    boosting_selection: pd.DataFrame,
):
    return fit_selected_gradient_boosting(synthetic_model_data, boosting_selection)


def test_cross_validation_covers_candidates_and_chronological_folds(
    synthetic_model_data: pd.DataFrame,
) -> None:
    results = cross_validate_gradient_boosting(
        synthetic_model_data,
        learning_rates=(0.05, 0.1),
        n_estimators_values=(20,),
        max_depths=(1, 2),
    )

    assert len(results) == 2 * 1 * 2 * 4
    assert results["train_end_year"].lt(results["validation_start_year"]).all()
    assert results["pr_auc"].between(0, 1).all()
    assert results["roc_auc"].between(0, 1).all()


def test_selection_returns_one_competitive_ensemble(
    boosting_selection: pd.DataFrame,
) -> None:
    selected = boosting_selection.iloc[0]

    assert len(boosting_selection) == 1
    assert selected["selection_rule"] == "one_standard_error_simplest_boosting"
    assert selected["mean_pr_auc"] >= selected["eligibility_threshold"]
    assert selected["folds"] == 4


def test_validation_values_cannot_change_final_boosting(
    synthetic_model_data: pd.DataFrame,
    boosting_selection: pd.DataFrame,
    fitted_boosting,
) -> None:
    modified = synthetic_model_data.copy()
    validation = modified["sample_period"].eq("validation")
    modified.loc[validation, MODEL_FEATURES] = 1_000_000
    refitted = fit_selected_gradient_boosting(modified, boosting_selection)

    original = fitted_boosting.named_steps["classifier"].feature_importances_
    refitted_importance = refitted.named_steps["classifier"].feature_importances_
    np.testing.assert_allclose(original, refitted_importance)


def test_feature_importance_covers_all_features(fitted_boosting) -> None:
    importance = build_gradient_boosting_feature_importance(fitted_boosting)

    assert len(importance) == len(MODEL_FEATURES)
    assert set(importance["feature"]) == set(MODEL_FEATURES)
    assert importance["importance"].sum() == pytest.approx(1.0)


def test_validation_predictions_are_valid_probabilities(
    synthetic_model_data: pd.DataFrame,
    fitted_boosting,
) -> None:
    predictions = create_gradient_boosting_validation_predictions(
        synthetic_model_data,
        fitted_boosting,
    )

    expected_rows = synthetic_model_data["sample_period"].eq("validation").sum()
    assert len(predictions) == expected_rows
    assert predictions["model"].eq(MODEL_NAME).all()
    assert predictions["predicted_probability"].between(0, 1).all()


def test_importance_plot_returns_a_matplotlib_figure(fitted_boosting) -> None:
    importance = build_gradient_boosting_feature_importance(fitted_boosting)
    figure = plot_gradient_boosting_feature_importance(importance, top_n=10)

    assert len(figure.axes) == 1
    plt.close(figure)
