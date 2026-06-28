"""Create future-safe development and test samples for bankruptcy prediction.

This module separates the firm-year feature table by fiscal year. Standard random
cross-validation is not used because it would allow later economic periods to influence
models evaluated on earlier years.

Main sample periods:
    - Training: 1999-2011
    - Validation: 2012-2014
    - Final test: 2015-2018

The training sample also receives four expanding-window folds. Each fold uses all
available history up to a fixed year and evaluates only on subsequent years. These folds
will later select model hyperparameters without touching validation or test observations.

Outputs:
    - A model dataset with an ordered ``sample_period`` column.
    - Summary tables for the main temporal periods and expanding-window folds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN


@dataclass(frozen=True)
class SamplePeriod:
    """Define one chronological segment of the empirical analysis."""

    name: str
    start_year: int
    end_year: int


@dataclass(frozen=True)
class ExpandingFold:
    """Define training and validation boundaries for one temporal fold."""

    name: str
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int


SAMPLE_PERIODS = (
    SamplePeriod("train", 1999, 2011),
    SamplePeriod("validation", 2012, 2014),
    SamplePeriod("test", 2015, 2018),
)

EXPANDING_FOLDS = (
    ExpandingFold("fold_1", 1999, 2004, 2005, 2006),
    ExpandingFold("fold_2", 1999, 2006, 2007, 2008),
    ExpandingFold("fold_3", 1999, 2008, 2009, 2010),
    ExpandingFold("fold_4", 1999, 2010, 2011, 2011),
)

PERIOD_ORDER = tuple(period.name for period in SAMPLE_PERIODS)


def assign_sample_periods(features: pd.DataFrame) -> pd.DataFrame:
    """Attach an ordered train, validation, or test label to every observation.

    Parameters
    ----------
    features:
        Model feature table containing company, fiscal-year, and target columns.

    Returns
    -------
    pandas.DataFrame
        A chronologically sorted copy with an ordered ``sample_period`` category.

    Raises
    ------
    ValueError
        If identifiers are missing, company-year keys are duplicated, or a fiscal year
        falls outside the pre-specified empirical design.

    """
    required_columns = {COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(features.columns)
    if missing_columns:
        raise ValueError(f"Required split columns are missing: {sorted(missing_columns)}")
    if features.duplicated([COMPANY_COLUMN, YEAR_COLUMN]).any():
        raise ValueError("Company-year keys must be unique before temporal splitting.")

    data = features.copy()
    sample_period = pd.Series(pd.NA, index=data.index, dtype="string")
    for period in SAMPLE_PERIODS:
        in_period = data[YEAR_COLUMN].between(period.start_year, period.end_year)
        sample_period.loc[in_period] = period.name

    if sample_period.isna().any():
        invalid_years = sorted(data.loc[sample_period.isna(), YEAR_COLUMN].unique())
        raise ValueError(f"Fiscal years outside the empirical design: {invalid_years}")

    data["sample_period"] = pd.Categorical(
        sample_period,
        categories=PERIOD_ORDER,
        ordered=True,
    )
    return data.sort_values([YEAR_COLUMN, COMPANY_COLUMN], ignore_index=True)


def summarize_sample_periods(model_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize years, firms, observations, and events in each main period."""
    required_columns = {COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN, "sample_period"}
    missing_columns = required_columns.difference(model_data.columns)
    if missing_columns:
        raise ValueError(f"Cannot summarize periods; columns are missing: {missing_columns}")

    summary = (
        model_data.groupby("sample_period", observed=True)
        .agg(
            first_year=(YEAR_COLUMN, "min"),
            last_year=(YEAR_COLUMN, "max"),
            observations=(TARGET_COLUMN, "size"),
            companies=(COMPANY_COLUMN, "nunique"),
            bankruptcies=(TARGET_COLUMN, "sum"),
            event_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )
    summary["bankruptcies"] = summary["bankruptcies"].astype(int)
    return summary


def make_expanding_window_splits(
    years: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return scikit-learn-compatible indices for expanding-window validation.

    Parameters
    ----------
    years:
        Fiscal years aligned positionally with a training feature matrix.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray]]
        Training and validation positional indices for each fold.

    """
    year_values = years.to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in EXPANDING_FOLDS:
        train_indices = np.flatnonzero(
            (year_values >= fold.train_start) & (year_values <= fold.train_end)
        )
        validation_indices = np.flatnonzero(
            (year_values >= fold.validation_start) & (year_values <= fold.validation_end)
        )
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(f"Insufficient observations for {fold.name}.")
        if year_values[train_indices].max() >= year_values[validation_indices].min():
            raise ValueError(f"Temporal leakage detected in {fold.name}.")
        splits.append((train_indices, validation_indices))
    return splits


def summarize_expanding_folds(model_data: pd.DataFrame) -> pd.DataFrame:
    """Return an auditable table of observations and events in every fold."""
    rows: list[dict[str, int | str]] = []
    for fold in EXPANDING_FOLDS:
        train = model_data[YEAR_COLUMN].between(fold.train_start, fold.train_end)
        validation = model_data[YEAR_COLUMN].between(
            fold.validation_start, fold.validation_end
        )
        rows.append(
            {
                **asdict(fold),
                "train_observations": int(train.sum()),
                "train_bankruptcies": int(model_data.loc[train, TARGET_COLUMN].sum()),
                "validation_observations": int(validation.sum()),
                "validation_bankruptcies": int(
                    model_data.loc[validation, TARGET_COLUMN].sum()
                ),
            }
        )
    return pd.DataFrame(rows)

