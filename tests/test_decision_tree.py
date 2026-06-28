from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bankruptcy_risk.decision_tree import (
    MODEL_NAME,
    build_tree_feature_importance,
    create_tree_validation_predictions,
    cross_validate_pruned_tree,
    fit_selected_pruned_tree,
    plot_selected_tree,
    select_pruned_tree,
)
from bankruptcy_risk.features import MODEL_FEATURES
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
        (data["debt_to_assets"] > 0.7)
        & (data["return_on_assets"] < -0.3)
    ) | (data["signed_log_total_assets"] < -1.5)
    data[TARGET_COLUMN] = np.where(event_signal, 1, rng.binomial(1, 0.03, len(data)))
    data["company_id"] = [f"company_{index // 4}" for index in range(len(data))]
    data[YEAR_COLUMN] = years
    data["sample_period"] = np.where(years <= 2011, "train", "validation")
    return data


@pytest.fixture(scope="module")
def tree_selection(synthetic_model_data: pd.DataFrame) -> pd.DataFrame:
    cv_results = cross_validate_pruned_tree(
        synthetic_model_data,
        max_depths=(3, 5),
        min_samples_leaf_values=(20,),
        ccp_alphas=(0.0, 0.0001),
    )
    return select_pruned_tree(cv_results)


@pytest.fixture(scope="module")
def fitted_tree(
    synthetic_model_data: pd.DataFrame,
    tree_selection: pd.DataFrame,
):
    return fit_selected_pruned_tree(synthetic_model_data, tree_selection)


def test_cross_validation_covers_candidates_and_chronological_folds(
    synthetic_model_data: pd.DataFrame,
) -> None:
    results = cross_validate_pruned_tree(
        synthetic_model_data,
        max_depths=(3, 5),
        min_samples_leaf_values=(20,),
        ccp_alphas=(0.0, 0.0001),
    )

    assert len(results) == 2 * 1 * 2 * 4
    assert results["train_end_year"].lt(results["validation_start_year"]).all()
    assert results["pr_auc"].between(0, 1).all()
    assert results["roc_auc"].between(0, 1).all()


def test_one_standard_error_rule_returns_one_competitive_tree(
    tree_selection: pd.DataFrame,
) -> None:
    selected = tree_selection.iloc[0]

    assert len(tree_selection) == 1
    assert selected["selection_rule"] == "one_standard_error_simplest_tree"
    assert selected["mean_pr_auc"] >= selected["eligibility_threshold"]
    assert selected["folds"] == 4


def test_one_standard_error_rule_prefers_simpler_eligible_candidate() -> None:
    complex_scores = [0.10, 0.14, 0.10, 0.14]
    simple_scores = [0.11, 0.11, 0.11, 0.11]
    rows = []
    for fold, score in enumerate(complex_scores, start=1):
        rows.append(
            {
                "model": MODEL_NAME,
                "max_depth": 5,
                "min_samples_leaf": 20,
                "ccp_alpha": 0.0,
                "fitted_tree_depth": 5,
                "fitted_leaf_count": 10,
                "pr_auc": score,
                "roc_auc": 0.80,
                "fold": f"fold_{fold}",
            }
        )
    for fold, score in enumerate(simple_scores, start=1):
        rows.append(
            {
                "model": MODEL_NAME,
                "max_depth": 3,
                "min_samples_leaf": 100,
                "ccp_alpha": 0.0001,
                "fitted_tree_depth": 3,
                "fitted_leaf_count": 4,
                "pr_auc": score,
                "roc_auc": 0.79,
                "fold": f"fold_{fold}",
            }
        )

    selected = select_pruned_tree(pd.DataFrame(rows)).iloc[0]

    assert selected["max_depth"] == 3
    assert selected["mean_fitted_leaf_count"] == 4


def test_validation_values_cannot_change_final_tree(
    synthetic_model_data: pd.DataFrame,
    tree_selection: pd.DataFrame,
    fitted_tree,
) -> None:
    modified = synthetic_model_data.copy()
    validation = modified["sample_period"].eq("validation")
    modified.loc[validation, MODEL_FEATURES] = 1_000_000
    refitted = fit_selected_pruned_tree(modified, tree_selection)

    original_tree = fitted_tree.named_steps["classifier"].tree_
    refitted_tree = refitted.named_steps["classifier"].tree_
    np.testing.assert_allclose(original_tree.threshold, refitted_tree.threshold)
    np.testing.assert_array_equal(original_tree.feature, refitted_tree.feature)


def test_feature_importance_covers_all_model_features(fitted_tree) -> None:
    importance = build_tree_feature_importance(fitted_tree)

    assert len(importance) == len(MODEL_FEATURES)
    assert set(importance["feature"]) == set(MODEL_FEATURES)
    assert importance["importance"].sum() == pytest.approx(1.0)
    assert importance["rank"].tolist() == list(range(1, len(MODEL_FEATURES) + 1))


def test_validation_predictions_are_valid_probabilities(
    synthetic_model_data: pd.DataFrame,
    fitted_tree,
) -> None:
    predictions = create_tree_validation_predictions(synthetic_model_data, fitted_tree)

    expected_rows = synthetic_model_data["sample_period"].eq("validation").sum()
    assert len(predictions) == expected_rows
    assert predictions["model"].eq(MODEL_NAME).all()
    assert predictions["predicted_probability"].between(0, 1).all()


def test_tree_plot_returns_a_matplotlib_figure(fitted_tree) -> None:
    figure = plot_selected_tree(fitted_tree, displayed_depth=2)

    assert len(figure.axes) == 1
    plt.close(figure)
