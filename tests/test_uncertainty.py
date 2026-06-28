from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bankruptcy_risk.evaluation import SUBSTANTIVE_MODELS
from bankruptcy_risk.target import TARGET_COLUMN, YEAR_COLUMN
from bankruptcy_risk.uncertainty import (
    METRICS,
    build_bootstrap_intervals,
    build_pairwise_differences,
    calculate_point_metrics,
    draw_company_cluster_indices,
    generate_clustered_bootstrap_replicates,
    plot_bootstrap_metric_intervals,
    plot_pairwise_differences,
)


def _make_synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    companies = np.repeat([f"company_{index}" for index in range(100)], 3)
    years = np.tile([2012, 2013, 2014], 100)
    target = np.zeros(len(companies), dtype=int)
    target[rng.choice(len(target), size=30, replace=False)] = 1
    signal = np.clip(0.06 + 0.65 * target + rng.normal(0, 0.15, len(target)), 0, 1)
    prediction_tables = []
    threshold_rows = []
    for index, model_name in enumerate(SUBSTANTIVE_MODELS):
        probabilities = np.clip(
            signal + rng.normal(0, 0.02 + 0.01 * index, len(target)),
            0.0001,
            0.9999,
        )
        prediction_tables.append(
            pd.DataFrame(
                {
                    "company_id": companies,
                    YEAR_COLUMN: years,
                    TARGET_COLUMN: target,
                    "model": model_name,
                    "predicted_probability": probabilities,
                }
            )
        )
        threshold_rows.append({"model": model_name, "selected_threshold": 0.35})
    return pd.concat(prediction_tables, ignore_index=True), pd.DataFrame(threshold_rows)


def test_cluster_draw_keeps_all_rows_for_each_sampled_company() -> None:
    company_ids = pd.Series(["A", "A", "B", "C", "C", "C"])
    sampled = draw_company_cluster_indices(company_ids, np.random.default_rng(7))
    sampled_ids = company_ids.iloc[sampled].value_counts()

    assert sampled_ids.get("A", 0) % 2 == 0
    assert sampled_ids.get("B", 0) % 1 == 0
    assert sampled_ids.get("C", 0) % 3 == 0


def test_bootstrap_replicates_are_paired_and_reproducible() -> None:
    predictions, thresholds = _make_synthetic_inputs()
    first = generate_clustered_bootstrap_replicates(
        predictions,
        thresholds,
        n_bootstrap=50,
        random_seed=12,
    )
    second = generate_clustered_bootstrap_replicates(
        predictions,
        thresholds,
        n_bootstrap=50,
        random_seed=12,
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 50 * len(SUBSTANTIVE_MODELS)
    assert first.groupby("bootstrap_id")["model"].nunique().eq(len(SUBSTANTIVE_MODELS)).all()


def test_bootstrap_intervals_cover_all_models_and_metrics() -> None:
    predictions, thresholds = _make_synthetic_inputs()
    point_metrics = calculate_point_metrics(predictions, thresholds)
    replicates = generate_clustered_bootstrap_replicates(
        predictions,
        thresholds,
        n_bootstrap=50,
    )
    intervals = build_bootstrap_intervals(point_metrics, replicates)

    assert len(intervals) == len(SUBSTANTIVE_MODELS) * len(METRICS)
    assert intervals["confidence_interval_lower"].le(
        intervals["confidence_interval_upper"]
    ).all()
    assert intervals["bootstrap_replicates"].eq(50).all()
    assert intervals["resampling_unit"].eq("company").all()


def test_pairwise_differences_use_same_bootstrap_replicates() -> None:
    predictions, thresholds = _make_synthetic_inputs()
    point_metrics = calculate_point_metrics(predictions, thresholds)
    replicates = generate_clustered_bootstrap_replicates(
        predictions,
        thresholds,
        n_bootstrap=50,
    )
    pairwise = build_pairwise_differences(point_metrics, replicates)

    expected_rows = (len(SUBSTANTIVE_MODELS) - 1) * len(METRICS)
    assert len(pairwise) == expected_rows
    assert pairwise["bootstrap_replicates"].eq(50).all()
    assert pairwise["probability_reference_better"].between(0, 1).all()


def test_uncertainty_plots_return_matplotlib_figures() -> None:
    predictions, thresholds = _make_synthetic_inputs()
    point_metrics = calculate_point_metrics(predictions, thresholds)
    replicates = generate_clustered_bootstrap_replicates(
        predictions,
        thresholds,
        n_bootstrap=20,
    )
    intervals = build_bootstrap_intervals(point_metrics, replicates)
    pairwise = build_pairwise_differences(point_metrics, replicates)
    figures = (
        plot_bootstrap_metric_intervals(intervals),
        plot_pairwise_differences(pairwise),
    )

    assert [len(figure.axes) for figure in figures] == [3, 3]
    for figure in figures:
        plt.close(figure)
