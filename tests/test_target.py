from __future__ import annotations

import pandas as pd
import pytest

from bankruptcy_risk.config import RAW_DATA_PATH
from bankruptcy_risk.target import (
    ACCOUNTING_COLUMN_NAMES,
    COMPANY_COLUMN,
    PUBLISHED_EVENT_COUNTS,
    TARGET_COLUMN,
    YEAR_COLUMN,
    construct_bankruptcy_panel,
)


@pytest.fixture(scope="module")
def raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture(scope="module")
def panel(raw_data: pd.DataFrame) -> pd.DataFrame:
    return construct_bankruptcy_panel(raw_data)


def test_target_contains_609_bankruptcy_events(panel: pd.DataFrame) -> None:
    assert int(panel[TARGET_COLUMN].sum()) == 609
    assert panel[TARGET_COLUMN].dtype == "int8"


def test_annual_events_match_published_counts(panel: pd.DataFrame) -> None:
    observed = panel.groupby(YEAR_COLUMN)[TARGET_COLUMN].sum().astype(int).to_dict()

    assert observed == PUBLISHED_EVENT_COUNTS


def test_only_final_observation_of_failed_company_is_positive(panel: pd.DataFrame) -> None:
    failed = panel.loc[panel["status_label"].eq("failed")]
    positive_year = failed.loc[failed[TARGET_COLUMN].eq(1)].set_index(COMPANY_COLUMN)[YEAR_COLUMN]
    final_year = failed.groupby(COMPANY_COLUMN)[YEAR_COLUMN].max()

    pd.testing.assert_series_equal(
        positive_year.sort_index(),
        final_year.sort_index(),
        check_names=False,
    )
    assert failed.groupby(COMPANY_COLUMN)[TARGET_COLUMN].sum().eq(1).all()


def test_alive_companies_never_receive_positive_target(panel: pd.DataFrame) -> None:
    alive_targets = panel.loc[panel["status_label"].eq("alive"), TARGET_COLUMN]

    assert alive_targets.eq(0).all()


def test_panel_uses_readable_accounting_names(panel: pd.DataFrame) -> None:
    assert set(ACCOUNTING_COLUMN_NAMES.values()).issubset(panel.columns)
    assert not set(ACCOUNTING_COLUMN_NAMES).intersection(panel.columns)


def test_panel_is_sorted_by_company_and_year(panel: pd.DataFrame) -> None:
    expected = panel.sort_values([COMPANY_COLUMN, YEAR_COLUMN], ignore_index=True)

    pd.testing.assert_frame_equal(panel, expected)


def test_inconsistent_company_status_is_rejected(raw_data: pd.DataFrame) -> None:
    modified = raw_data.copy()
    modified.loc[modified.index[0], "status_label"] = "failed"

    with pytest.raises(ValueError, match="consistent eventual status"):
        construct_bankruptcy_panel(modified)


def test_incorrect_event_counts_are_rejected(raw_data: pd.DataFrame) -> None:
    incorrect_counts = {**PUBLISHED_EVENT_COUNTS, 2018: 35}

    with pytest.raises(ValueError, match="published annual counts"):
        construct_bankruptcy_panel(raw_data, expected_event_counts=incorrect_counts)

