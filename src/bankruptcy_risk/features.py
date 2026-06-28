"""Engineer financially meaningful predictors for bankruptcy modelling.

This module converts the prepared firm-year accounting panel into a model feature
table. It retains the company identifier, fiscal year, and one-year-ahead target while
removing ``status_label`` so a company's eventual outcome cannot leak into a model.

Inputs:
    - ``data/interim/bankruptcy_panel.parquet``

Outputs:
    - Sign-preserving logarithms of the 18 accounting amounts.
    - Profitability, liquidity, leverage, efficiency, and valuation ratios.
    - A feature dictionary describing every model input and formula.

Important design choices:
    - Zero denominators produce missing values rather than arbitrary replacements.
    - Missing-value imputation and outlier clipping are deliberately deferred to a
      later preprocessing pipeline fitted only on training observations.
    - No feature uses information from a later fiscal year.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from bankruptcy_risk.target import (
    ACCOUNTING_COLUMN_NAMES,
    COMPANY_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
)

ACCOUNTING_FEATURES = tuple(ACCOUNTING_COLUMN_NAMES.values())
SIGNED_LOG_FEATURES = tuple(f"signed_log_{feature}" for feature in ACCOUNTING_FEATURES)


@dataclass(frozen=True)
class FeatureDefinition:
    """Describe one engineered predictor in the model feature table."""

    feature: str
    category: str
    description: str
    formula: str


RATIO_DEFINITIONS = (
    FeatureDefinition(
        "current_ratio",
        "liquidity",
        "Short-term assets available per unit of short-term liabilities.",
        "current_assets / current_liabilities",
    ),
    FeatureDefinition(
        "working_capital_to_assets",
        "liquidity",
        "Net short-term resources relative to the asset base.",
        "(current_assets - current_liabilities) / total_assets",
    ),
    FeatureDefinition(
        "debt_to_assets",
        "leverage",
        "Share of total assets financed by liabilities.",
        "total_liabilities / total_assets",
    ),
    FeatureDefinition(
        "long_term_debt_to_assets",
        "leverage",
        "Long-term debt relative to the asset base.",
        "long_term_debt / total_assets",
    ),
    FeatureDefinition(
        "return_on_assets",
        "profitability",
        "Net income generated per unit of total assets.",
        "net_income / total_assets",
    ),
    FeatureDefinition(
        "ebitda_to_assets",
        "profitability",
        "Operating earnings before depreciation relative to assets.",
        "ebitda / total_assets",
    ),
    FeatureDefinition(
        "ebit_to_assets",
        "profitability",
        "Operating earnings after depreciation relative to assets.",
        "ebit / total_assets",
    ),
    FeatureDefinition(
        "retained_earnings_to_assets",
        "profitability",
        "Accumulated retained earnings relative to assets.",
        "retained_earnings / total_assets",
    ),
    FeatureDefinition(
        "asset_turnover",
        "efficiency",
        "Sales generated per unit of total assets.",
        "net_sales / total_assets",
    ),
    FeatureDefinition(
        "net_profit_margin",
        "profitability",
        "Net income retained from each unit of revenue.",
        "net_income / total_revenue",
    ),
    FeatureDefinition(
        "gross_margin",
        "profitability",
        "Gross profit retained from each unit of net sales.",
        "gross_profit / net_sales",
    ),
    FeatureDefinition(
        "market_value_to_liabilities",
        "valuation",
        "Market value available relative to total liabilities.",
        "market_value / total_liabilities",
    ),
    FeatureDefinition(
        "inventory_to_current_assets",
        "liquidity",
        "Share of current assets held as inventory.",
        "inventory / current_assets",
    ),
    FeatureDefinition(
        "receivables_to_sales",
        "efficiency",
        "Receivables outstanding relative to net sales.",
        "total_receivables / net_sales",
    ),
)

RATIO_FEATURES = tuple(definition.feature for definition in RATIO_DEFINITIONS)
MODEL_FEATURES = (*SIGNED_LOG_FEATURES, *RATIO_FEATURES)
INTERPRETABLE_FEATURES = (
    "signed_log_total_assets",
    "working_capital_to_assets",
    "debt_to_assets",
    "return_on_assets",
    "retained_earnings_to_assets",
    "asset_turnover",
    "market_value_to_liabilities",
)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    tolerance: float = 1e-12,
) -> pd.Series:
    """Divide two series and return missing values for near-zero denominators.

    Parameters
    ----------
    numerator:
        Values appearing above the ratio line.
    denominator:
        Values appearing below the ratio line.
    tolerance:
        Smallest denominator magnitude treated as numerically valid.

    Returns
    -------
    pandas.Series
        Floating-point ratios without positive or negative infinity.

    """
    valid_denominator = denominator.astype(float).where(denominator.abs() > tolerance)
    ratio = numerator.astype(float).div(valid_denominator)
    return ratio.replace([np.inf, -np.inf], np.nan)


def signed_log1p(values: pd.Series) -> pd.Series:
    """Compress heavy tails while preserving zero and the sign of losses."""
    numeric_values = values.astype(float)
    return np.sign(numeric_values) * np.log1p(np.abs(numeric_values))


def build_feature_dictionary() -> pd.DataFrame:
    """Return definitions for all signed-log and financial-ratio features."""
    log_definitions = [
        FeatureDefinition(
            feature=f"signed_log_{feature}",
            category="accounting amount",
            description=f"Sign-preserving logarithm of {feature.replace('_', ' ')}.",
            formula=f"sign({feature}) * log(1 + abs({feature}))",
        )
        for feature in ACCOUNTING_FEATURES
    ]
    definitions = [*log_definitions, *RATIO_DEFINITIONS]
    return pd.DataFrame(asdict(definition) for definition in definitions)


def engineer_model_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Create the complete model feature table from the audited panel.

    Parameters
    ----------
    panel:
        Firm-year panel produced by :func:`construct_bankruptcy_panel`.

    Returns
    -------
    pandas.DataFrame
        Company and year identifiers, the prediction target, and all engineered
        predictors. The eventual ``status_label`` is intentionally excluded.

    Raises
    ------
    ValueError
        If required accounting columns are missing or company-year keys are duplicated.

    """
    required_columns = {COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN, *ACCOUNTING_FEATURES}
    missing_columns = required_columns.difference(panel.columns)
    if missing_columns:
        raise ValueError(f"Required panel columns are missing: {sorted(missing_columns)}")
    if panel.duplicated([COMPANY_COLUMN, YEAR_COLUMN]).any():
        raise ValueError("Company-year keys must remain unique during feature engineering.")

    features = panel.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN]].copy()
    for accounting_feature in ACCOUNTING_FEATURES:
        features[f"signed_log_{accounting_feature}"] = signed_log1p(
            panel[accounting_feature]
        )

    features["current_ratio"] = safe_divide(
        panel["current_assets"], panel["current_liabilities"]
    )
    features["working_capital_to_assets"] = safe_divide(
        panel["current_assets"] - panel["current_liabilities"], panel["total_assets"]
    )
    features["debt_to_assets"] = safe_divide(
        panel["total_liabilities"], panel["total_assets"]
    )
    features["long_term_debt_to_assets"] = safe_divide(
        panel["long_term_debt"], panel["total_assets"]
    )
    features["return_on_assets"] = safe_divide(
        panel["net_income"], panel["total_assets"]
    )
    features["ebitda_to_assets"] = safe_divide(panel["ebitda"], panel["total_assets"])
    features["ebit_to_assets"] = safe_divide(panel["ebit"], panel["total_assets"])
    features["retained_earnings_to_assets"] = safe_divide(
        panel["retained_earnings"], panel["total_assets"]
    )
    features["asset_turnover"] = safe_divide(panel["net_sales"], panel["total_assets"])
    features["net_profit_margin"] = safe_divide(
        panel["net_income"], panel["total_revenue"]
    )
    features["gross_margin"] = safe_divide(panel["gross_profit"], panel["net_sales"])
    features["market_value_to_liabilities"] = safe_divide(
        panel["market_value"], panel["total_liabilities"]
    )
    features["inventory_to_current_assets"] = safe_divide(
        panel["inventory"], panel["current_assets"]
    )
    features["receivables_to_sales"] = safe_divide(
        panel["total_receivables"], panel["net_sales"]
    )

    if np.isinf(features.loc[:, MODEL_FEATURES].to_numpy()).any():
        raise ValueError("Engineered features must not contain infinite values.")
    return features.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN, *MODEL_FEATURES]]
