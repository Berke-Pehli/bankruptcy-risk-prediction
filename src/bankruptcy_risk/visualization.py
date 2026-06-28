"""Create publication-ready figures for exploratory bankruptcy analysis.

The plotting functions accept prepared data frames and return Matplotlib figures rather
than writing files directly. Pytask controls output paths and persistence, which keeps
visual logic testable and separates computation from filesystem side effects.

Figures:
    - Annual firm coverage and bankruptcy prevalence.
    - Class balance across train, validation, and test periods.
    - Training-only distributions for selected financial ratios.
    - Training-only correlation heatmap for all engineered ratios.

Target-conditioned figures call the training-sample selector internally, ensuring that
validation and test outcomes remain unseen during exploratory feature assessment.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from bankruptcy_risk.exploration import (
    SELECTED_DISTRIBUTION_RATIOS,
    build_training_ratio_correlation,
    get_training_observations,
)
from bankruptcy_risk.target import TARGET_COLUMN, YEAR_COLUMN

PRIMARY_COLOR = "#1F4E79"
ACCENT_COLOR = "#C55A11"
HEALTHY_COLOR = "#5B9BD5"
BANKRUPT_COLOR = "#C00000"


def _apply_figure_style() -> None:
    """Apply a restrained style shared by all project figures."""
    sns.set_theme(style="whitegrid", context="notebook")


def plot_annual_overview(annual_overview: pd.DataFrame) -> Figure:
    """Plot annual company coverage, bankruptcy events, and event rates."""
    _apply_figure_style()
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    years = annual_overview[YEAR_COLUMN].astype(int)

    axes[0].plot(
        years,
        annual_overview["companies"],
        color=PRIMARY_COLOR,
        marker="o",
        linewidth=2,
    )
    axes[0].set_title("Annual Company Coverage")
    axes[0].set_ylabel("Companies")

    axes[1].bar(
        years,
        annual_overview["bankruptcies"],
        color=ACCENT_COLOR,
        alpha=0.8,
        label="Bankruptcy events",
    )
    axes[1].set_ylabel("Bankruptcy events")
    axes[1].set_xlabel("Fiscal year of accounting information")
    axes[1].set_title("One-Year-Ahead Bankruptcy Events and Event Rate")

    rate_axis = axes[1].twinx()
    rate_axis.plot(
        years,
        annual_overview["event_rate"] * 100,
        color=PRIMARY_COLOR,
        marker="o",
        linewidth=2,
        label="Event rate",
    )
    rate_axis.set_ylabel("Event rate (%)")
    rate_axis.grid(False)

    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = rate_axis.get_legend_handles_labels()
    axes[1].legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")
    displayed_years = years.iloc[::2]
    axes[1].set_xticks(displayed_years)
    axes[1].set_xticklabels(displayed_years.astype(str))
    figure.tight_layout()
    return figure


def plot_period_event_rates(period_summary: pd.DataFrame) -> Figure:
    """Compare bankruptcy prevalence across the pre-specified sample periods."""
    _apply_figure_style()
    figure, axis = plt.subplots(figsize=(8, 5))
    rates = period_summary["event_rate"] * 100
    period_labels = period_summary["sample_period"].str.title()
    bars = axis.bar(period_labels, rates, color=PRIMARY_COLOR, width=0.6)
    axis.bar_label(bars, labels=[f"{rate:.2f}%" for rate in rates], padding=3)
    axis.set_title("One-Year-Ahead Bankruptcy Rate by Sample Period")
    axis.set_xlabel("Sample period")
    axis.set_ylabel("Bankruptcy rate (%)")
    axis.set_ylim(0, rates.max() * 1.25)
    figure.tight_layout()
    return figure


def plot_training_ratio_distributions(model_data: pd.DataFrame) -> Figure:
    """Compare selected ratio distributions by training-sample outcome."""
    _apply_figure_style()
    training = get_training_observations(model_data)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    for axis, feature in zip(axes.flat, SELECTED_DISTRIBUTION_RATIOS, strict=True):
        plot_data = training.loc[:, [TARGET_COLUMN, feature]].copy()
        lower, upper = plot_data[feature].quantile([0.01, 0.99])
        plot_data[feature] = plot_data[feature].clip(lower, upper)
        plot_data["outcome"] = plot_data[TARGET_COLUMN].map(
            {0: "No bankruptcy", 1: "Bankruptcy next year"}
        )
        sns.boxplot(
            data=plot_data,
            x="outcome",
            y=feature,
            hue="outcome",
            palette=[HEALTHY_COLOR, BANKRUPT_COLOR],
            legend=False,
            showfliers=False,
            ax=axis,
        )
        axis.set_title(feature.replace("_", " ").title())
        axis.set_xlabel("")
        axis.set_ylabel("Training-sample value (1st-99th percentile)")

    figure.suptitle("Selected Financial Ratios by Bankruptcy Outcome", y=1.02)
    figure.tight_layout()
    return figure


def plot_training_ratio_correlation(model_data: pd.DataFrame) -> Figure:
    """Plot the training-only correlation matrix for engineered ratios."""
    _apply_figure_style()
    correlation = build_training_ratio_correlation(model_data)
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Pearson correlation"},
        ax=axis,
    )
    readable_labels = [label.replace("_", " ").title() for label in correlation.columns]
    axis.set_xticklabels(readable_labels, rotation=45, ha="right")
    axis.set_yticklabels(readable_labels, rotation=0)
    axis.set_title("Training-Sample Correlations among Financial Ratios")
    figure.tight_layout()
    return figure
