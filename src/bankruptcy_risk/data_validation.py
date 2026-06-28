"""Validate the canonical bankruptcy dataset before transformation.

The checks in this module form a data contract. They make source-file changes fail
early, before a modified schema or corrupted observation can affect model results.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

ACCOUNTING_COLUMNS = tuple(f"X{number}" for number in range(1, 19))
EXPECTED_COLUMNS = ("company_name", "status_label", "year", *ACCOUNTING_COLUMNS)
EXPECTED_SHA256 = "cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e"


class DataValidationError(ValueError):
    """Indicate that the raw dataset violates a required project assumption."""


@dataclass(frozen=True)
class RawDataContract:
    """Expected properties of the unmodified Kaggle CSV."""

    rows: int = 78_682
    columns: int = 21
    companies: int = 8_971
    failed_companies: int = 609
    first_year: int = 1999
    last_year: int = 2018
    sha256: str = EXPECTED_SHA256


DEFAULT_CONTRACT = RawDataContract()


@dataclass(frozen=True)
class DataValidationReport:
    """Auditable facts recorded after the raw data pass validation."""

    rows: int
    columns: int
    companies: int
    failed_companies: int
    first_year: int
    last_year: int
    missing_values: int
    duplicate_rows: int
    duplicate_company_years: int
    sha256: str

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-serializable representation of the report."""
        return asdict(self)


def calculate_sha256(path: Path) -> str:
    """Calculate a file digest without loading the full CSV into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file_checksum(path: Path, expected_sha256: str) -> str:
    """Verify file identity and return the observed SHA-256 digest."""
    observed_sha256 = calculate_sha256(path)
    if observed_sha256 != expected_sha256:
        raise DataValidationError(
            "Raw-data checksum mismatch. The canonical CSV may have been modified."
        )
    return observed_sha256


def validate_raw_dataframe(
    data: pd.DataFrame,
    contract: RawDataContract = DEFAULT_CONTRACT,
) -> None:
    """Check the schema, identifiers, labels, years, and accounting values.

    Parameters
    ----------
    data:
        DataFrame read directly from the canonical CSV.
    contract:
        Expected properties against which the data are checked.

    Raises
    ------
    DataValidationError
        If any property required by the empirical design is violated.

    """
    if tuple(data.columns) != EXPECTED_COLUMNS:
        raise DataValidationError("Raw-data columns or their order do not match the contract.")
    if data.shape != (contract.rows, contract.columns):
        raise DataValidationError(
            f"Expected shape {(contract.rows, contract.columns)}, found {data.shape}."
        )
    if data.isna().any().any():
        raise DataValidationError("The canonical raw data must not contain missing values.")
    if data.duplicated().any():
        raise DataValidationError("The canonical raw data contain duplicate rows.")
    if data.duplicated(["company_name", "year"]).any():
        raise DataValidationError("Each company-year key must be unique.")
    if set(data["status_label"].unique()) != {"alive", "failed"}:
        raise DataValidationError("status_label must contain exactly 'alive' and 'failed'.")
    if not is_integer_dtype(data["year"]):
        raise DataValidationError("year must be stored as an integer column.")
    if (int(data["year"].min()), int(data["year"].max())) != (
        contract.first_year,
        contract.last_year,
    ):
        raise DataValidationError("Fiscal-year coverage does not match the contract.")

    company_status_counts = data.groupby("company_name", observed=True)["status_label"].nunique()
    if not company_status_counts.eq(1).all():
        raise DataValidationError("Each company must have one consistent eventual status label.")
    if data["company_name"].nunique() != contract.companies:
        raise DataValidationError("The number of unique companies does not match the contract.")
    failed_companies = data.loc[data["status_label"].eq("failed"), "company_name"].nunique()
    if failed_companies != contract.failed_companies:
        raise DataValidationError("The number of failed companies does not match the contract.")

    non_numeric = [column for column in ACCOUNTING_COLUMNS if not is_numeric_dtype(data[column])]
    if non_numeric:
        raise DataValidationError(f"Accounting columns must be numeric: {non_numeric}")
    if not np.isfinite(data.loc[:, ACCOUNTING_COLUMNS].to_numpy()).all():
        raise DataValidationError("Accounting columns must contain only finite values.")


def load_and_validate_raw_data(
    path: Path,
    contract: RawDataContract = DEFAULT_CONTRACT,
) -> tuple[pd.DataFrame, DataValidationReport]:
    """Load the canonical CSV, enforce its contract, and return an audit report."""
    observed_sha256 = validate_file_checksum(path, contract.sha256)
    data = pd.read_csv(path)
    validate_raw_dataframe(data, contract)

    failed_companies = data.loc[data["status_label"].eq("failed"), "company_name"].nunique()
    report = DataValidationReport(
        rows=len(data),
        columns=len(data.columns),
        companies=data["company_name"].nunique(),
        failed_companies=failed_companies,
        first_year=int(data["year"].min()),
        last_year=int(data["year"].max()),
        missing_values=int(data.isna().sum().sum()),
        duplicate_rows=int(data.duplicated().sum()),
        duplicate_company_years=int(data.duplicated(["company_name", "year"]).sum()),
        sha256=observed_sha256,
    )
    return data, report
