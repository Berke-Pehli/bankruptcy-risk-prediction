from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bankruptcy_risk.evaluation import SUBSTANTIVE_MODELS
from bankruptcy_risk.features import MODEL_FEATURES
from bankruptcy_risk.final_evaluation import (
    build_final_model_audit,
    build_final_test_calibration,
    build_final_test_intervals,
    build_validation_test_comparison,
    create_final_test_predictions,
    evaluate_final_test_predictions,
    fit_final_champion,
    freeze_final_specification,
    generate_final_test_bootstrap,
    plot_final_test_confusion_matrix,
    plot_final_test_diagnostics,
)
from bankruptcy_risk.target import TARGET_COLUMN, YEAR_COLUMN


def _make_inputs():
    rng = np.random.default_rng(42)
    observations = 900
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
    data[YEAR_COLUMN] = np.select(
        [np.arange(observations) < 450, np.arange(observations) < 650],
        [2011, 2013],
        default=2016,
    )
    data["sample_period"] = np.select(
        [np.arange(observations) < 450, np.arange(observations) < 650],
        ["train", "validation"],
        default="test",
    )
    validation_metrics = pd.DataFrame(
        {
            "model": SUBSTANTIVE_MODELS,
            "pr_auc": [0.10, 0.18, 0.17, 0.12, 0.25, 0.16],
        }
    )
    thresholds = pd.DataFrame(
        {
            "model": SUBSTANTIVE_MODELS,
            "selected_threshold": [0.02, 0.03, 0.03, 0.04, 0.05, 0.04],
            "selection_rule": "maximum_validation_f2",
        }
    )
    forest_selection = pd.DataFrame(
        [{"max_depth": 5, "min_samples_leaf": 5, "max_features": 0.5}]
    )
    return data, validation_metrics, thresholds, forest_selection


def test_frozen_specification_uses_validation_choices() -> None:
    _, validation_metrics, thresholds, forest_selection = _make_inputs()
    specification = freeze_final_specification(
        validation_metrics,
        thresholds,
        forest_selection,
        n_estimators=30,
    )

    assert specification.model == "random_forest"
    assert specification.selected_threshold == 0.05
    assert specification.max_depth == 5
    assert specification.n_estimators == 30


def test_test_values_cannot_change_refitted_model() -> None:
    data, validation_metrics, thresholds, forest_selection = _make_inputs()
    specification = freeze_final_specification(
        validation_metrics,
        thresholds,
        forest_selection,
        n_estimators=30,
    )
    fitted = fit_final_champion(data, specification)
    modified = data.copy()
    test = modified["sample_period"].eq("test")
    modified.loc[test, MODEL_FEATURES] = 1_000_000
    modified.loc[test, TARGET_COLUMN] = 1 - modified.loc[test, TARGET_COLUMN]
    refitted = fit_final_champion(modified, specification)

    np.testing.assert_allclose(
        fitted.named_steps["classifier"].feature_importances_,
        refitted.named_steps["classifier"].feature_importances_,
    )


def test_final_predictions_and_metrics_use_only_test_period() -> None:
    data, validation_metrics, thresholds, forest_selection = _make_inputs()
    specification = freeze_final_specification(
        validation_metrics,
        thresholds,
        forest_selection,
        n_estimators=30,
    )
    fitted = fit_final_champion(data, specification)
    predictions = create_final_test_predictions(data, fitted, specification)
    metrics = evaluate_final_test_predictions(predictions, specification)

    assert len(predictions) == data["sample_period"].eq("test").sum()
    assert set(predictions[YEAR_COLUMN]) == {2016}
    assert predictions["predicted_probability"].between(0, 1).all()
    assert metrics["threshold_source"].tolist() == [
        "default_0.50_reference",
        "frozen_validation_f2",
    ]
    assert metrics["test_used_for_selection"].eq(False).all()


def test_final_bootstrap_and_intervals_are_company_clustered() -> None:
    data, validation_metrics, thresholds, forest_selection = _make_inputs()
    specification = freeze_final_specification(
        validation_metrics,
        thresholds,
        forest_selection,
        n_estimators=30,
    )
    fitted = fit_final_champion(data, specification)
    predictions = create_final_test_predictions(data, fitted, specification)
    metrics = evaluate_final_test_predictions(predictions, specification)
    bootstrap = generate_final_test_bootstrap(
        predictions,
        specification,
        n_bootstrap=50,
    )
    intervals = build_final_test_intervals(metrics, bootstrap)

    assert len(bootstrap) == 50
    assert len(intervals) == 6
    assert intervals["bootstrap_replicates"].eq(50).all()
    assert intervals["resampling_unit"].eq("company").all()


def test_final_audit_comparison_and_figures_are_complete() -> None:
    data, validation_metrics, thresholds, forest_selection = _make_inputs()
    specification = freeze_final_specification(
        validation_metrics,
        thresholds,
        forest_selection,
        n_estimators=30,
    )
    fitted = fit_final_champion(data, specification)
    predictions = create_final_test_predictions(data, fitted, specification)
    metrics = evaluate_final_test_predictions(predictions, specification)
    audit = build_final_model_audit(data, specification)
    calibration = build_final_test_calibration(predictions, n_bins=5)
    validation_optimized = metrics.loc[
        metrics["threshold_source"].eq("frozen_validation_f2")
    ].copy()
    comparison = build_validation_test_comparison(validation_optimized, metrics)
    figures = (
        plot_final_test_diagnostics(predictions, calibration),
        plot_final_test_confusion_matrix(predictions, specification),
    )

    assert audit.loc[0, "test_used_for_model_selection"] == False  # noqa: E712
    assert comparison["evaluation_period"].tolist() == [
        "validation_2012_2014",
        "final_test_2015_2018",
    ]
    assert [len(figure.axes) for figure in figures] == [3, 1]
    for figure in figures:
        plt.close(figure)
