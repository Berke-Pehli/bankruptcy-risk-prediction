from __future__ import annotations

import pandas as pd
import pytest

from bankruptcy_risk.config import RAW_DATA_PATH
from bankruptcy_risk.features import engineer_model_features
from bankruptcy_risk.splitting import (
    EXPANDING_FOLDS,
    PERIOD_ORDER,
    assign_sample_periods,
    make_expanding_window_splits,
    summarize_sample_periods,
)
from bankruptcy_risk.target import TARGET_COLUMN, YEAR_COLUMN, construct_bankruptcy_panel


@pytest.fixture(scope="module")
def model_data() -> pd.DataFrame:
    raw_data = pd.read_csv(RAW_DATA_PATH)
    panel = construct_bankruptcy_panel(raw_data)
    features = engineer_model_features(panel)
    return assign_sample_periods(features)


def test_all_observations_receive_one_sample_period(model_data: pd.DataFrame) -> None:
    assert model_data["sample_period"].notna().all()
    assert tuple(model_data["sample_period"].cat.categories) == PERIOD_ORDER


def test_main_period_boundaries_are_chronological(model_data: pd.DataFrame) -> None:
    years = model_data.groupby("sample_period", observed=True)[YEAR_COLUMN].agg(["min", "max"])

    assert tuple(years.loc["train"]) == (1999, 2011)
    assert tuple(years.loc["validation"]) == (2012, 2014)
    assert tuple(years.loc["test"]) == (2015, 2018)


def test_main_period_observation_and_event_counts(model_data: pd.DataFrame) -> None:
    summary = summarize_sample_periods(model_data).set_index("sample_period")

    assert summary["observations"].to_dict() == {
        "train": 55_927,
        "validation": 10_473,
        "test": 12_282,
    }
    assert summary["bankruptcies"].to_dict() == {
        "train": 403,
        "validation": 87,
        "test": 119,
    }


def test_expanding_folds_never_train_on_future_years(model_data: pd.DataFrame) -> None:
    training_data = model_data.loc[model_data["sample_period"].eq("train")].reset_index(
        drop=True
    )
    splits = make_expanding_window_splits(training_data[YEAR_COLUMN])

    assert len(splits) == len(EXPANDING_FOLDS)
    for train_indices, validation_indices in splits:
        train_years = training_data.iloc[train_indices][YEAR_COLUMN]
        validation_years = training_data.iloc[validation_indices][YEAR_COLUMN]
        assert train_years.max() < validation_years.min()


def test_expanding_folds_contain_bankruptcies_in_both_samples(
    model_data: pd.DataFrame,
) -> None:
    training_data = model_data.loc[model_data["sample_period"].eq("train")].reset_index(
        drop=True
    )

    for train_indices, validation_indices in make_expanding_window_splits(
        training_data[YEAR_COLUMN]
    ):
        assert training_data.iloc[train_indices][TARGET_COLUMN].sum() > 0
        assert training_data.iloc[validation_indices][TARGET_COLUMN].sum() > 0


def test_year_outside_design_is_rejected(model_data: pd.DataFrame) -> None:
    modified = model_data.drop(columns="sample_period").copy()
    modified.loc[0, YEAR_COLUMN] = 2019

    with pytest.raises(ValueError, match="outside the empirical design"):
        assign_sample_periods(modified)

