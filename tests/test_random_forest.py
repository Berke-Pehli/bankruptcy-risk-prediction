from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bankruptcy_risk.features import MODEL_FEATURES
from bankruptcy_risk.random_forest import (
    MODEL_NAME,
    build_random_forest_feature_importance,
    build_random_forest_oob_diagnostics,
    create_random_forest_validation_predictions,
    cross_validate_random_forest,
    fit_selected_random_forest,
    plot_random_forest_feature_importance,
    select_random_forest,
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
def forest_selection(synthetic_model_data: pd.DataFrame) -> pd.DataFrame:
    cv_results = cross_validate_random_forest(
        synthetic_model_data,
        max_depths=(4, 8),
        min_samples_leaf_values=(10,),
        max_features_values=(0.3,),
        n_estimators=30,
    )
    return select_random_forest(cv_results)


@pytest.fixture(scope="module")
def fitted_forest(
    synthetic_model_data: pd.DataFrame,
    forest_selection: pd.DataFrame,
):
    return fit_selected_random_forest(
        synthetic_model_data,
        forest_selection,
        n_estimators=100,
    )


def test_cross_validation_covers_candidates_and_chronological_folds(
    synthetic_model_data: pd.DataFrame,
) -> None:
    results = cross_validate_random_forest(
        synthetic_model_data,
        max_depths=(4, 8),
        min_samples_leaf_values=(10,),
        max_features_values=(0.3,),
        n_estimators=30,
    )

    assert len(results) == 2 * 1 * 1 * 4
    assert results["train_end_year"].lt(results["validation_start_year"]).all()
    assert results["pr_auc"].between(0, 1).all()
    assert results["roc_auc"].between(0, 1).all()


def test_selection_returns_one_competitive_forest(
    forest_selection: pd.DataFrame,
) -> None:
    selected = forest_selection.iloc[0]

    assert len(forest_selection) == 1
    assert selected["selection_rule"] == "one_standard_error_simplest_forest"
    assert selected["mean_pr_auc"] >= selected["eligibility_threshold"]
    assert selected["folds"] == 4


def test_validation_values_cannot_change_final_forest(
    synthetic_model_data: pd.DataFrame,
    forest_selection: pd.DataFrame,
    fitted_forest,
) -> None:
    modified = synthetic_model_data.copy()
    validation = modified["sample_period"].eq("validation")
    modified.loc[validation, MODEL_FEATURES] = 1_000_000
    refitted = fit_selected_random_forest(
        modified,
        forest_selection,
        n_estimators=100,
    )

    original_importance = fitted_forest.named_steps["classifier"].feature_importances_
    refitted_importance = refitted.named_steps["classifier"].feature_importances_
    np.testing.assert_allclose(original_importance, refitted_importance)


def test_oob_diagnostics_are_finite_probabilistic_metrics(
    synthetic_model_data: pd.DataFrame,
    fitted_forest,
) -> None:
    diagnostics = build_random_forest_oob_diagnostics(
        synthetic_model_data,
        fitted_forest,
    ).iloc[0]

    assert diagnostics["model"] == MODEL_NAME
    assert 0 <= diagnostics["oob_pr_auc"] <= 1
    assert 0 <= diagnostics["oob_roc_auc"] <= 1
    assert 0 <= diagnostics["oob_brier_score"] <= 1


def test_feature_importance_covers_all_features(fitted_forest) -> None:
    importance = build_random_forest_feature_importance(fitted_forest)

    assert len(importance) == len(MODEL_FEATURES)
    assert set(importance["feature"]) == set(MODEL_FEATURES)
    assert importance["importance"].sum() == pytest.approx(1.0)
    assert importance["importance_std_across_trees"].ge(0).all()


def test_validation_predictions_are_valid_probabilities(
    synthetic_model_data: pd.DataFrame,
    fitted_forest,
) -> None:
    predictions = create_random_forest_validation_predictions(
        synthetic_model_data,
        fitted_forest,
    )

    expected_rows = synthetic_model_data["sample_period"].eq("validation").sum()
    assert len(predictions) == expected_rows
    assert predictions["model"].eq(MODEL_NAME).all()
    assert predictions["predicted_probability"].between(0, 1).all()


def test_importance_plot_returns_a_matplotlib_figure(fitted_forest) -> None:
    importance = build_random_forest_feature_importance(fitted_forest)
    figure = plot_random_forest_feature_importance(importance, top_n=10)

    assert len(figure.axes) == 1
    plt.close(figure)
