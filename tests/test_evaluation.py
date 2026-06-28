from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bankruptcy_risk.evaluation import (
    CALIBRATION_MODELS,
    MODEL_ORDER,
    SUBSTANTIVE_MODELS,
    build_calibration_table,
    build_threshold_curves,
    combine_validation_predictions,
    evaluate_predictions,
    plot_calibration_curves,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_selected_confusion_matrices,
    select_f2_thresholds,
)
from bankruptcy_risk.target import TARGET_COLUMN, YEAR_COLUMN


@pytest.fixture(scope="module")
def combined_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    observations = 200
    target = np.zeros(observations, dtype=int)
    target[rng.choice(observations, size=24, replace=False)] = 1
    signal = np.clip(0.08 + 0.65 * target + rng.normal(0, 0.16, observations), 0, 1)
    tables = []
    for index, model_name in enumerate(MODEL_ORDER):
        if model_name == "majority_non_bankrupt":
            probabilities = np.zeros(observations)
        elif model_name == "training_prevalence":
            probabilities = np.full(observations, target.mean())
        else:
            noise = rng.normal(0, 0.02 + index * 0.002, observations)
            probabilities = np.clip(signal + noise, 0.0001, 0.9999)
        tables.append(
            pd.DataFrame(
                {
                    "company_id": [f"company_{row}" for row in range(observations)],
                    YEAR_COLUMN: np.repeat([2012, 2013], observations // 2),
                    TARGET_COLUMN: target,
                    "model": model_name,
                    "predicted_probability": probabilities,
                    "predicted_class": (probabilities >= 0.5).astype(int),
                }
            )
        )
    return combine_validation_predictions(tables)


def test_combined_predictions_cover_same_rows_for_every_model(
    combined_predictions: pd.DataFrame,
) -> None:
    counts = combined_predictions.groupby("model").size()

    assert set(counts.index) == set(MODEL_ORDER)
    assert counts.nunique() == 1


def test_inconsistent_model_target_is_rejected(combined_predictions: pd.DataFrame) -> None:
    tables = [
        combined_predictions.loc[combined_predictions["model"].eq(model)].copy()
        for model in MODEL_ORDER
    ]
    tables[-1].loc[tables[-1].index[0], TARGET_COLUMN] ^= 1

    with pytest.raises(ValueError, match="keys or outcomes differ"):
        combine_validation_predictions(tables)


def test_default_metrics_cover_all_models(combined_predictions: pd.DataFrame) -> None:
    metrics = evaluate_predictions(combined_predictions)

    assert metrics["model"].tolist() == list(MODEL_ORDER)
    assert metrics["threshold"].eq(0.5).all()
    assert metrics["roc_auc"].between(0, 1).all()
    assert metrics["pr_auc"].between(0, 1).all()
    assert metrics["brier_score"].between(0, 1).all()


def test_selected_thresholds_maximize_f2(combined_predictions: pd.DataFrame) -> None:
    curves = build_threshold_curves(combined_predictions)
    selected = select_f2_thresholds(curves)

    assert selected["model"].tolist() == list(SUBSTANTIVE_MODELS)
    for row in selected.itertuples(index=False):
        model_curve = curves.loc[curves["model"].eq(row.model)]
        assert row.validation_f2_score == pytest.approx(model_curve["f2_score"].max())


def test_optimized_metrics_use_selected_thresholds(
    combined_predictions: pd.DataFrame,
) -> None:
    selected = select_f2_thresholds(build_threshold_curves(combined_predictions))
    threshold_map = selected.set_index("model")["selected_threshold"].to_dict()
    metrics = evaluate_predictions(
        combined_predictions,
        thresholds=threshold_map,
        models=SUBSTANTIVE_MODELS,
        threshold_source="maximum_validation_f2",
    )

    assert metrics["model"].tolist() == list(SUBSTANTIVE_MODELS)
    assert metrics["threshold_source"].eq("maximum_validation_f2").all()
    assert metrics["recall"].between(0, 1).all()


def test_calibration_table_covers_requested_models(
    combined_predictions: pd.DataFrame,
) -> None:
    calibration = build_calibration_table(combined_predictions, n_bins=5)

    assert set(calibration["model"]) == set(CALIBRATION_MODELS)
    assert calibration["mean_predicted_probability"].between(0, 1).all()
    assert calibration["observed_event_rate"].between(0, 1).all()


def test_evaluation_plots_return_matplotlib_figures(
    combined_predictions: pd.DataFrame,
) -> None:
    selected = select_f2_thresholds(build_threshold_curves(combined_predictions))
    calibration = build_calibration_table(combined_predictions, n_bins=5)
    figures = (
        plot_precision_recall_curves(combined_predictions),
        plot_roc_curves(combined_predictions),
        plot_calibration_curves(calibration),
        plot_selected_confusion_matrices(combined_predictions, selected),
    )

    assert [len(figure.axes) for figure in figures] == [1, 1, 1, 6]
    for figure in figures:
        plt.close(figure)
