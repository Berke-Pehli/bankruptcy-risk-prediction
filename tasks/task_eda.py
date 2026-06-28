"""Generate reproducible exploratory tables and figures.

Inputs:
    - ``data/processed/model_dataset.parquet``
    - The pre-computed temporal split summary.
    - Exploration and visualization source modules tracked by Pytask.

Outputs:
    - Annual bankruptcy overview and training-ratio summary tables.
    - Four publication-ready PNG figures.

All target-conditioned ratio analysis is restricted to the training period. Validation
and test observations appear only in descriptive coverage and class-balance summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product

from bankruptcy_risk.config import (
    ANNUAL_OVERVIEW_FIGURE_PATH,
    ANNUAL_OVERVIEW_TABLE_PATH,
    EXPLORATION_MODULE_PATH,
    MODEL_DATASET_PATH,
    PERIOD_BALANCE_FIGURE_PATH,
    RATIO_CORRELATION_FIGURE_PATH,
    RATIO_DISTRIBUTION_FIGURE_PATH,
    TEMPORAL_SPLIT_SUMMARY_PATH,
    TRAINING_RATIO_SUMMARY_PATH,
    VISUALIZATION_MODULE_PATH,
)
from bankruptcy_risk.exploration import build_annual_overview, build_training_ratio_summary
from bankruptcy_risk.visualization import (
    plot_annual_overview,
    plot_period_event_rates,
    plot_training_ratio_correlation,
    plot_training_ratio_distributions,
)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Save a figure consistently and release its Matplotlib resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def task_write_annual_overview(
    model_data_path: Path = MODEL_DATASET_PATH,
    exploration_module_path: Path = EXPLORATION_MODULE_PATH,
    table_path: Annotated[Path, Product] = ANNUAL_OVERVIEW_TABLE_PATH,
) -> None:
    """Write annual company counts, events, and event rates."""
    if not exploration_module_path.exists():
        raise FileNotFoundError("The exploration module must exist.")
    model_data = pd.read_parquet(model_data_path)
    overview = build_annual_overview(model_data)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    overview.to_csv(table_path, index=False)


def task_write_training_ratio_summary(
    model_data_path: Path = MODEL_DATASET_PATH,
    exploration_module_path: Path = EXPLORATION_MODULE_PATH,
    table_path: Annotated[Path, Product] = TRAINING_RATIO_SUMMARY_PATH,
) -> None:
    """Write training-only descriptive statistics by bankruptcy outcome."""
    if not exploration_module_path.exists():
        raise FileNotFoundError("The exploration module must exist.")
    model_data = pd.read_parquet(model_data_path)
    summary = build_training_ratio_summary(model_data)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)


def task_plot_annual_overview(
    table_path: Path = ANNUAL_OVERVIEW_TABLE_PATH,
    visualization_module_path: Path = VISUALIZATION_MODULE_PATH,
    figure_path: Annotated[Path, Product] = ANNUAL_OVERVIEW_FIGURE_PATH,
) -> None:
    """Plot annual coverage, bankruptcy events, and event rates."""
    if not visualization_module_path.exists():
        raise FileNotFoundError("The visualization module must exist.")
    annual_overview = pd.read_csv(table_path)
    _save_figure(plot_annual_overview(annual_overview), figure_path)


def task_plot_period_event_rates(
    summary_path: Path = TEMPORAL_SPLIT_SUMMARY_PATH,
    visualization_module_path: Path = VISUALIZATION_MODULE_PATH,
    figure_path: Annotated[Path, Product] = PERIOD_BALANCE_FIGURE_PATH,
) -> None:
    """Plot class prevalence across train, validation, and test periods."""
    if not visualization_module_path.exists():
        raise FileNotFoundError("The visualization module must exist.")
    summary = pd.read_csv(summary_path)
    _save_figure(plot_period_event_rates(summary), figure_path)


def task_plot_training_ratio_distributions(
    model_data_path: Path = MODEL_DATASET_PATH,
    exploration_module_path: Path = EXPLORATION_MODULE_PATH,
    visualization_module_path: Path = VISUALIZATION_MODULE_PATH,
    figure_path: Annotated[Path, Product] = RATIO_DISTRIBUTION_FIGURE_PATH,
) -> None:
    """Plot selected training-only ratios by bankruptcy outcome."""
    if not exploration_module_path.exists() or not visualization_module_path.exists():
        raise FileNotFoundError("Exploration and visualization modules must exist.")
    model_data = pd.read_parquet(model_data_path)
    _save_figure(plot_training_ratio_distributions(model_data), figure_path)


def task_plot_training_ratio_correlation(
    model_data_path: Path = MODEL_DATASET_PATH,
    exploration_module_path: Path = EXPLORATION_MODULE_PATH,
    visualization_module_path: Path = VISUALIZATION_MODULE_PATH,
    figure_path: Annotated[Path, Product] = RATIO_CORRELATION_FIGURE_PATH,
) -> None:
    """Plot the training-only correlation matrix for financial ratios."""
    if not exploration_module_path.exists() or not visualization_module_path.exists():
        raise FileNotFoundError("Exploration and visualization modules must exist.")
    model_data = pd.read_parquet(model_data_path)
    _save_figure(plot_training_ratio_correlation(model_data), figure_path)

