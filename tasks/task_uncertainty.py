"""Estimate company-clustered uncertainty for validation model comparisons.

Inputs:
    - Combined validation predictions.
    - Validation-selected F2 thresholds.
    - Uncertainty and configuration source modules tracked by Pytask.

Outputs:
    - Point metrics, 1,000 bootstrap replicates, percentile intervals, and paired Random
      Forest comparison tables.
    - Metric-interval and pairwise-difference figures.

Companies are sampled with replacement and all their validation-year observations are
kept together. Test-period observations remain untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product

from bankruptcy_risk.config import (
    COMBINED_VALIDATION_PREDICTIONS_PATH,
    CONFIG_MODULE_PATH,
    UNCERTAINTY_MODULE_PATH,
    VALIDATION_BOOTSTRAP_INTERVALS_FIGURE_PATH,
    VALIDATION_BOOTSTRAP_INTERVALS_PATH,
    VALIDATION_BOOTSTRAP_PAIRWISE_FIGURE_PATH,
    VALIDATION_BOOTSTRAP_PAIRWISE_PATH,
    VALIDATION_BOOTSTRAP_POINT_METRICS_PATH,
    VALIDATION_BOOTSTRAP_REPLICATES_PATH,
    VALIDATION_SELECTED_THRESHOLDS_PATH,
)
from bankruptcy_risk.uncertainty import (
    build_bootstrap_intervals,
    build_pairwise_differences,
    calculate_point_metrics,
    generate_clustered_bootstrap_replicates,
    plot_bootstrap_metric_intervals,
    plot_pairwise_differences,
)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one uncertainty figure and release its Matplotlib resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def task_bootstrap_validation_uncertainty(
    predictions_path: Path = COMBINED_VALIDATION_PREDICTIONS_PATH,
    thresholds_path: Path = VALIDATION_SELECTED_THRESHOLDS_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    uncertainty_module_path: Path = UNCERTAINTY_MODULE_PATH,
    point_metrics_path: Annotated[
        Path, Product
    ] = VALIDATION_BOOTSTRAP_POINT_METRICS_PATH,
    replicates_path: Annotated[Path, Product] = VALIDATION_BOOTSTRAP_REPLICATES_PATH,
    intervals_path: Annotated[Path, Product] = VALIDATION_BOOTSTRAP_INTERVALS_PATH,
    pairwise_path: Annotated[Path, Product] = VALIDATION_BOOTSTRAP_PAIRWISE_PATH,
    intervals_figure_path: Annotated[
        Path, Product
    ] = VALIDATION_BOOTSTRAP_INTERVALS_FIGURE_PATH,
    pairwise_figure_path: Annotated[
        Path, Product
    ] = VALIDATION_BOOTSTRAP_PAIRWISE_FIGURE_PATH,
) -> None:
    """Write clustered-bootstrap estimates, comparisons, and figures."""
    if not config_module_path.exists() or not uncertainty_module_path.exists():
        raise FileNotFoundError("Uncertainty source modules must exist.")
    predictions = pd.read_csv(predictions_path)
    thresholds = pd.read_csv(thresholds_path)
    point_metrics = calculate_point_metrics(predictions, thresholds)
    replicates = generate_clustered_bootstrap_replicates(predictions, thresholds)
    intervals = build_bootstrap_intervals(point_metrics, replicates)
    pairwise = build_pairwise_differences(point_metrics, replicates)

    table_outputs = (
        (point_metrics, point_metrics_path),
        (replicates, replicates_path),
        (intervals, intervals_path),
        (pairwise, pairwise_path),
    )
    for table, path in table_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    _save_figure(plot_bootstrap_metric_intervals(intervals), intervals_figure_path)
    _save_figure(plot_pairwise_differences(pairwise), pairwise_figure_path)
