from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bankruptcy_risk.config import RAW_DATA_PATH
from bankruptcy_risk.data_validation import (
    ACCOUNTING_COLUMNS,
    EXPECTED_COLUMNS,
    EXPECTED_SHA256,
    DataValidationError,
    RawDataContract,
    calculate_sha256,
    load_and_validate_raw_data,
    validate_file_checksum,
    validate_raw_dataframe,
)


@pytest.fixture(scope="module")
def canonical_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


def test_canonical_file_satisfies_contract() -> None:
    data, report = load_and_validate_raw_data(RAW_DATA_PATH)

    assert data.shape == (78_682, 21)
    assert report.companies == 8_971
    assert report.failed_companies == 609
    assert report.sha256 == EXPECTED_SHA256


def test_expected_columns_are_explicit_and_ordered() -> None:
    assert EXPECTED_COLUMNS == (
        "company_name",
        "status_label",
        "year",
        *ACCOUNTING_COLUMNS,
    )


def test_missing_column_is_rejected(canonical_data: pd.DataFrame) -> None:
    modified = canonical_data.drop(columns="X18")

    with pytest.raises(DataValidationError, match="columns or their order"):
        validate_raw_dataframe(modified)


def test_duplicate_company_year_is_rejected(canonical_data: pd.DataFrame) -> None:
    modified = canonical_data.copy()
    modified.loc[1, ["company_name", "year"]] = modified.loc[
        0, ["company_name", "year"]
    ].to_numpy()

    with pytest.raises(DataValidationError, match="company-year"):
        validate_raw_dataframe(modified)


def test_non_finite_accounting_value_is_rejected(canonical_data: pd.DataFrame) -> None:
    modified = canonical_data.copy()
    modified.loc[0, "X1"] = np.inf

    with pytest.raises(DataValidationError, match="finite values"):
        validate_raw_dataframe(modified)


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    temporary_file = tmp_path / "example.csv"
    temporary_file.write_text("changed content\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="checksum mismatch"):
        validate_file_checksum(temporary_file, expected_sha256="not-the-observed-digest")


def test_calculate_sha256_is_deterministic(tmp_path: Path) -> None:
    temporary_file = tmp_path / "example.txt"
    temporary_file.write_text("bankruptcy-risk\n", encoding="utf-8")

    assert calculate_sha256(temporary_file) == calculate_sha256(temporary_file)


def test_contract_defaults_capture_canonical_shape() -> None:
    contract = RawDataContract()

    assert contract.rows == 78_682
    assert contract.columns == 21

