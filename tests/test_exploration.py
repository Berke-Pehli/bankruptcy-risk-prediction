from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from bankruptcy_risk.config import RAW_DATA_PATH
from bankruptcy_risk.exploration import (
    build_annual_overview,
    build_training_ratio_correlation,
    build_training_ratio_summary,
)
from bankruptcy_risk.features import RATIO_FEATURES, engineer_model_features
from bankruptcy_risk.splitting import assign_sample_periods, summarize_sample_periods
from bankruptcy_risk.target import construct_bankruptcy_panel
from bankruptcy_risk.visualization import (
    plot_annual_overview,
    plot_period_event_rates,
    plot_training_ratio_correlation,
    plot_training_ratio_distributions,
)


@pytest.fixture(scope="module")
def model_data() -> pd.DataFrame:
    raw_data = pd.read_csv(RAW_DATA_PATH)
    panel = construct_bankruptcy_panel(raw_data)
    features = engineer_model_features(panel)
    return assign_sample_periods(features)


def test_annual_overview_covers_full_panel(model_data: pd.DataFrame) -> None:
    overview = build_annual_overview(model_data)

    assert len(overview) == 20
    assert int(overview["observations"].sum()) == 78_682
    assert int(overview["bankruptcies"].sum()) == 609


def test_ratio_summary_uses_training_observations_only(model_data: pd.DataFrame) -> None:
    summary = build_training_ratio_summary(model_data)
    counts_per_feature = summary.groupby("feature")["observations"].sum()

    assert len(summary["feature"].unique()) == len(RATIO_FEATURES)
    assert counts_per_feature.eq(55_927).all()


def test_validation_changes_do_not_affect_training_summary(model_data: pd.DataFrame) -> None:
    baseline = build_training_ratio_summary(model_data)
    modified = model_data.copy()
    validation = modified["sample_period"].eq("validation")
    modified.loc[validation, RATIO_FEATURES] = 1_000_000

    pd.testing.assert_frame_equal(build_training_ratio_summary(modified), baseline)


def test_training_correlation_has_expected_shape(model_data: pd.DataFrame) -> None:
    correlation = build_training_ratio_correlation(model_data)

    assert correlation.shape == (len(RATIO_FEATURES), len(RATIO_FEATURES))
    assert correlation.index.tolist() == correlation.columns.tolist()


def test_exploratory_plot_functions_return_figures(model_data: pd.DataFrame) -> None:
    annual = build_annual_overview(model_data)
    periods = summarize_sample_periods(model_data)
    figures = [
        plot_annual_overview(annual),
        plot_period_event_rates(periods),
        plot_training_ratio_distributions(model_data),
        plot_training_ratio_correlation(model_data),
    ]

    try:
        assert all(isinstance(figure, Figure) for figure in figures)
        assert len(figures[0].axes) == 3
        assert len(figures[2].axes) == 4
    finally:
        for figure in figures:
            plt.close(figure)

