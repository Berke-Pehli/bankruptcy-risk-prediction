"""Prepare descriptive evidence for the bankruptcy prediction study.

This module creates compact tables used to understand the panel before fitting any
model. It separates two types of exploratory analysis:

Full-sample descriptive analysis:
    - Annual company coverage.
    - Bankruptcy counts and event rates.
    - Changes in class prevalence across the pre-specified sample periods.

Training-only predictive analysis:
    - Financial-ratio distributions by the one-year-ahead target.
    - Correlations among engineered ratios.

Using only training observations for target-conditioned analysis prevents validation or
test outcomes from influencing feature choices. The full sample is used only to describe
the dataset's historical coverage and pre-specified temporal partitions.

Inputs:
    - ``data/processed/model_dataset.parquet``

Outputs:
    - Annual overview and training-ratio summary tables.
    - Data frames consumed by the visualization tasks.
"""

from __future__ import annotations

import pandas as pd

from bankruptcy_risk.features import RATIO_FEATURES
from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN

SELECTED_DISTRIBUTION_RATIOS = (
    "return_on_assets",
    "debt_to_assets",
    "working_capital_to_assets",
    "retained_earnings_to_assets",
)


def get_training_observations(model_data: pd.DataFrame) -> pd.DataFrame:
    """Return an independent copy containing only pre-specified training years."""
    if "sample_period" not in model_data.columns:
        raise ValueError("model_data must contain the sample_period column.")
    training = model_data.loc[model_data["sample_period"].eq("train")].copy()
    if training.empty:
        raise ValueError("No training observations were found in model_data.")
    return training


def build_annual_overview(model_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize company coverage and bankruptcy prevalence by fiscal year."""
    required_columns = {COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(model_data.columns)
    if missing_columns:
        raise ValueError(f"Cannot build annual overview; missing columns: {missing_columns}")

    overview = (
        model_data.groupby(YEAR_COLUMN, observed=True)
        .agg(
            observations=(TARGET_COLUMN, "size"),
            companies=(COMPANY_COLUMN, "nunique"),
            bankruptcies=(TARGET_COLUMN, "sum"),
            event_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )
    overview["bankruptcies"] = overview["bankruptcies"].astype(int)
    return overview


def build_training_ratio_summary(model_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize every financial ratio by training-sample bankruptcy outcome.

    Returns
    -------
    pandas.DataFrame
        Tidy table containing observations, mean, standard deviation, quartiles, and
        median for each financial ratio and target class.

    """
    training = get_training_observations(model_data)
    missing_ratios = set(RATIO_FEATURES).difference(training.columns)
    if missing_ratios:
        raise ValueError(f"Financial ratios are missing: {sorted(missing_ratios)}")

    long_data = training.loc[:, [TARGET_COLUMN, *RATIO_FEATURES]].melt(
        id_vars=TARGET_COLUMN,
        var_name="feature",
        value_name="value",
    )
    summary = (
        long_data.groupby(["feature", TARGET_COLUMN], observed=True)["value"]
        .agg(
            observations="count",
            mean="mean",
            standard_deviation="std",
            first_quartile=lambda values: values.quantile(0.25),
            median="median",
            third_quartile=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    return summary


def build_training_ratio_correlation(model_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate ratio correlations using training observations only."""
    training = get_training_observations(model_data)
    return training.loc[:, RATIO_FEATURES].corr()

