from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bankruptcy_risk.evaluation import SUBSTANTIVE_MODELS
from bankruptcy_risk.features import INTERPRETABLE_FEATURES, MODEL_FEATURES
from bankruptcy_risk.interpretation import (
    build_cross_model_driver_importance,
    build_driver_consensus,
    calculate_champion_partial_dependence,
    calculate_champion_permutation_importance,
    plot_champion_partial_dependence,
    plot_champion_permutation_importance,
    plot_cross_model_driver_heatmap,
    select_validation_champion,
)
from bankruptcy_risk.random_forest import make_random_forest_pipeline
from bankruptcy_risk.target import TARGET_COLUMN, YEAR_COLUMN


@pytest.fixture(scope="module")
def synthetic_model_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    observations = 600
    data = pd.DataFrame(
        rng.normal(size=(observations, len(MODEL_FEATURES))),
        columns=MODEL_FEATURES,
    )
    event_signal = (
        (data["debt_to_assets"] > 0.5)
        & (data["return_on_assets"] < -0.2)
    ) | (data["market_value_to_liabilities"] < -1.2)
    data[TARGET_COLUMN] = np.where(event_signal, 1, rng.binomial(1, 0.04, observations))
    data["company_id"] = [f"company_{index // 3}" for index in range(observations)]
    data[YEAR_COLUMN] = np.where(np.arange(observations) < 450, 2011, 2012)
    data["sample_period"] = np.where(
        np.arange(observations) < 450,
        "train",
        "validation",
    )
    return data


@pytest.fixture(scope="module")
def fitted_random_forest(synthetic_model_data: pd.DataFrame):
    training = synthetic_model_data.loc[synthetic_model_data["sample_period"].eq("train")]
    model = make_random_forest_pipeline(
        max_depth=5,
        min_samples_leaf=5,
        max_features=0.5,
        n_estimators=40,
    )
    model.fit(training.loc[:, MODEL_FEATURES], training[TARGET_COLUMN])
    return model


def _make_fake_model_artifacts():
    interpretable = SimpleNamespace(
        result=SimpleNamespace(
            params=pd.Series(
                np.linspace(-0.6, 0.6, len(INTERPRETABLE_FEATURES)),
                index=INTERPRETABLE_FEATURES,
            )
        )
    )

    def linear_pipeline(scale: float):
        classifier = SimpleNamespace(
            coef_=np.asarray([np.linspace(-scale, scale, len(MODEL_FEATURES))])
        )
        return SimpleNamespace(named_steps={"classifier": classifier})

    def tree_pipeline(offset: float):
        values = np.arange(1, len(MODEL_FEATURES) + 1, dtype=float) + offset
        classifier = SimpleNamespace(feature_importances_=values / values.sum())
        return SimpleNamespace(named_steps={"classifier": classifier})

    regularized = {
        "ridge_logit": linear_pipeline(0.5),
        "lasso_logit": linear_pipeline(0.3),
    }
    return (
        interpretable,
        regularized,
        tree_pipeline(0),
        tree_pipeline(1),
        tree_pipeline(2),
    )


def test_champion_is_selected_by_substantive_pr_auc() -> None:
    metrics = pd.DataFrame(
        {
            "model": SUBSTANTIVE_MODELS,
            "pr_auc": [0.10, 0.20, 0.19, 0.12, 0.25, 0.18],
        }
    )

    assert select_validation_champion(metrics) == "random_forest"


def test_permutation_importance_covers_every_feature(
    synthetic_model_data: pd.DataFrame,
    fitted_random_forest,
) -> None:
    importance = calculate_champion_permutation_importance(
        "random_forest",
        fitted_random_forest,
        synthetic_model_data,
        n_repeats=2,
        max_samples=100,
    )

    assert len(importance) == len(MODEL_FEATURES)
    assert set(importance["feature"]) == set(MODEL_FEATURES)
    assert importance["rank"].tolist() == list(range(1, len(MODEL_FEATURES) + 1))


def test_partial_dependence_uses_leading_permutation_features(
    synthetic_model_data: pd.DataFrame,
    fitted_random_forest,
) -> None:
    importance = calculate_champion_permutation_importance(
        "random_forest",
        fitted_random_forest,
        synthetic_model_data,
        n_repeats=2,
        max_samples=100,
    )
    dependence = calculate_champion_partial_dependence(
        "random_forest",
        fitted_random_forest,
        synthetic_model_data,
        importance,
        n_features=2,
        grid_resolution=10,
        max_samples=100,
    )

    assert dependence["feature"].drop_duplicates().tolist() == importance.head(2)[
        "feature"
    ].tolist()
    assert dependence["average_predicted_probability"].between(0, 1).all()


def test_cross_model_importance_is_normalized_with_coverage() -> None:
    artifacts = _make_fake_model_artifacts()
    importance = build_cross_model_driver_importance(*artifacts)
    consensus = build_driver_consensus(importance)

    model_sums = importance.groupby("model")["normalized_importance"].sum()
    assert set(model_sums.index) == set(SUBSTANTIVE_MODELS)
    assert np.allclose(model_sums, 1.0)
    assert consensus["consensus_rank"].tolist() == list(range(1, len(consensus) + 1))
    assert consensus["model_coverage_fraction"].between(0, 1).all()


def test_interpretation_plots_return_matplotlib_figures(
    synthetic_model_data: pd.DataFrame,
    fitted_random_forest,
) -> None:
    permutation = calculate_champion_permutation_importance(
        "random_forest",
        fitted_random_forest,
        synthetic_model_data,
        n_repeats=2,
        max_samples=100,
    )
    dependence = calculate_champion_partial_dependence(
        "random_forest",
        fitted_random_forest,
        synthetic_model_data,
        permutation,
        n_features=4,
        grid_resolution=10,
        max_samples=100,
    )
    importance = build_cross_model_driver_importance(*_make_fake_model_artifacts())
    consensus = build_driver_consensus(importance)
    figures = (
        plot_champion_permutation_importance(permutation),
        plot_champion_partial_dependence(dependence),
        plot_cross_model_driver_heatmap(importance, consensus),
    )

    assert [len(figure.axes) for figure in figures] == [1, 4, 2]
    for figure in figures:
        plt.close(figure)
