"""Construct the one-year-ahead bankruptcy outcome and firm-year panel.

The source file repeats each company's eventual status across its full history. This
module converts that company-level label into the row-level prediction outcome defined
by the dataset paper: accounting information from the final observed fiscal year of a
failed company is used to predict its bankruptcy in the following year.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

COMPANY_COLUMN = "company_id"
YEAR_COLUMN = "fiscal_year"
TARGET_COLUMN = "bankrupt_next_year"

ACCOUNTING_COLUMN_NAMES = {
    "X1": "current_assets",
    "X2": "cost_of_goods_sold",
    "X3": "depreciation_and_amortization",
    "X4": "ebitda",
    "X5": "inventory",
    "X6": "net_income",
    "X7": "total_receivables",
    "X8": "market_value",
    "X9": "net_sales",
    "X10": "total_assets",
    "X11": "long_term_debt",
    "X12": "ebit",
    "X13": "gross_profit",
    "X14": "current_liabilities",
    "X15": "retained_earnings",
    "X16": "total_revenue",
    "X17": "total_liabilities",
    "X18": "total_operating_expenses",
}

# Keys are fiscal years containing the predictors. The corresponding bankruptcy
# occurs in the following calendar year, as reported in Table 1 of the source paper.
PUBLISHED_EVENT_COUNTS = {
    1999: 3,
    2000: 7,
    2001: 10,
    2002: 17,
    2003: 29,
    2004: 46,
    2005: 40,
    2006: 51,
    2007: 59,
    2008: 58,
    2009: 23,
    2010: 35,
    2011: 25,
    2012: 26,
    2013: 28,
    2014: 33,
    2015: 33,
    2016: 29,
    2017: 21,
    2018: 36,
}


def _validate_company_status(data: pd.DataFrame) -> None:
    """Ensure each company has one eventual status across its history."""
    status_counts = data.groupby(COMPANY_COLUMN, observed=True)["status_label"].nunique()
    if not status_counts.eq(1).all():
        raise ValueError("Each company must have one consistent eventual status label.")


def _validate_event_counts(
    panel: pd.DataFrame,
    expected_event_counts: Mapping[int, int],
) -> None:
    """Ensure the reconstructed events reproduce the published yearly totals."""
    observed = (
        panel.groupby(YEAR_COLUMN, observed=True)[TARGET_COLUMN].sum().astype(int).to_dict()
    )
    if observed != dict(expected_event_counts):
        raise ValueError(
            "Constructed bankruptcy events do not match the published annual counts."
        )


def construct_bankruptcy_panel(
    raw_data: pd.DataFrame,
    expected_event_counts: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Create a readable panel with a one-year-ahead bankruptcy target.

    Parameters
    ----------
    raw_data:
        Validated Kaggle data with one observation per company and fiscal year.
    expected_event_counts:
        Expected positive targets by predictor fiscal year. Supplying this argument
        explicitly is mainly useful when testing the function on another data extract.

    Returns
    -------
    pandas.DataFrame
        A panel ordered by company and fiscal year. Accounting columns have descriptive
        names, while ``status_label`` is retained solely for auditing target construction.

    Raises
    ------
    ValueError
        If company statuses are inconsistent or reconstructed events differ from the
        published dataset totals.

    """
    data = raw_data.rename(
        columns={
            "company_name": COMPANY_COLUMN,
            "year": YEAR_COLUMN,
            **ACCOUNTING_COLUMN_NAMES,
        }
    ).copy()
    data[YEAR_COLUMN] = data[YEAR_COLUMN].astype(int)
    _validate_company_status(data)

    final_observed_year = data.groupby(COMPANY_COLUMN, observed=True)[YEAR_COLUMN].transform(
        "max"
    )
    data[TARGET_COLUMN] = (
        data["status_label"].eq("failed") & data[YEAR_COLUMN].eq(final_observed_year)
    ).astype("int8")

    ordered_columns = [
        COMPANY_COLUMN,
        YEAR_COLUMN,
        "status_label",
        TARGET_COLUMN,
        *ACCOUNTING_COLUMN_NAMES.values(),
    ]
    panel = data.loc[:, ordered_columns].sort_values(
        [COMPANY_COLUMN, YEAR_COLUMN], ignore_index=True
    )
    if expected_event_counts is None:
        expected_event_counts = PUBLISHED_EVENT_COUNTS
    _validate_event_counts(panel, expected_event_counts)
    return panel
