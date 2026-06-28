from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankruptcy_risk.config import RAW_DATA_PATH
from bankruptcy_risk.features import (
    ACCOUNTING_FEATURES,
    MODEL_FEATURES,
    RATIO_FEATURES,
    SIGNED_LOG_FEATURES,
    build_feature_dictionary,
    engineer_model_features,
    safe_divide,
    signed_log1p,
)
from bankruptcy_risk.target import (
    COMPANY_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    construct_bankruptcy_panel,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    raw_data = pd.read_csv(RAW_DATA_PATH)
    return construct_bankruptcy_panel(raw_data)


@pytest.fixture(scope="module")
def model_features(panel: pd.DataFrame) -> pd.DataFrame:
    return engineer_model_features(panel)


def test_safe_divide_replaces_zero_denominator_with_missing_value() -> None:
    numerator = pd.Series([4.0, 2.0, -3.0])
    denominator = pd.Series([2.0, 0.0, -1.0])

    result = safe_divide(numerator, denominator)

    assert result.iloc[0] == 2.0
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 3.0


def test_signed_log_preserves_sign_and_zero() -> None:
    values = pd.Series([-9.0, 0.0, 9.0])

    result = signed_log1p(values)

    assert result.iloc[0] < 0
    assert result.iloc[1] == 0
    assert result.iloc[2] > 0
    assert abs(result.iloc[0]) == pytest.approx(result.iloc[2])


def test_feature_table_preserves_identifiers_and_target(
    panel: pd.DataFrame,
    model_features: pd.DataFrame,
) -> None:
    expected = panel.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN]]

    pd.testing.assert_frame_equal(model_features.loc[:, expected.columns], expected)


def test_eventual_status_is_excluded(model_features: pd.DataFrame) -> None:
    assert "status_label" not in model_features.columns


def test_all_declared_features_are_created(model_features: pd.DataFrame) -> None:
    assert set(MODEL_FEATURES).issubset(model_features.columns)
    assert len(SIGNED_LOG_FEATURES) == len(ACCOUNTING_FEATURES) == 18
    assert len(RATIO_FEATURES) == 14


def test_selected_ratios_match_accounting_definitions(
    panel: pd.DataFrame,
    model_features: pd.DataFrame,
) -> None:
    row_index = panel.index[
        panel["current_liabilities"].ne(0) & panel["total_assets"].ne(0)
    ][0]

    expected_current_ratio = (
        panel.loc[row_index, "current_assets"] / panel.loc[row_index, "current_liabilities"]
    )
    expected_return_on_assets = (
        panel.loc[row_index, "net_income"] / panel.loc[row_index, "total_assets"]
    )

    assert model_features.loc[row_index, "current_ratio"] == pytest.approx(
        expected_current_ratio
    )
    assert model_features.loc[row_index, "return_on_assets"] == pytest.approx(
        expected_return_on_assets
    )


def test_model_features_do_not_contain_infinity(model_features: pd.DataFrame) -> None:
    values = model_features.loc[:, MODEL_FEATURES].to_numpy()

    assert not np.isinf(values).any()


def test_feature_dictionary_covers_every_model_feature_once() -> None:
    dictionary = build_feature_dictionary()

    assert dictionary["feature"].is_unique
    assert set(dictionary["feature"]) == set(MODEL_FEATURES)
    assert dictionary[["category", "description", "formula"]].notna().all().all()


def test_missing_accounting_column_is_rejected(panel: pd.DataFrame) -> None:
    incomplete_panel = panel.drop(columns="total_assets")

    with pytest.raises(ValueError, match="Required panel columns are missing"):
        engineer_model_features(incomplete_panel)

