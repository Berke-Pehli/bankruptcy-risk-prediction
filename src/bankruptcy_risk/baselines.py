"""Create simple reference predictions for the validation period.

This module establishes how difficult the bankruptcy-classification problem is before
fitting a statistical or machine-learning model. The benchmarks use only the training
period and are evaluated on the untouched validation years.

Input:
    - ``data/processed/model_dataset.parquet`` with temporal sample labels.

Outputs:
    - Majority baseline: always predicts a non-bankrupt company and probability zero.
    - Prevalence baseline: assigns every company the bankruptcy rate observed in the
      1999-2011 training sample.

These deliberately simple predictions provide reference points for probability-based
metrics and classification metrics. They do not alter the observed class distribution
or create synthetic bankruptcy events.
"""

from __future__ import annotations

import pandas as pd

from bankruptcy_risk.target import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN

PREDICTION_COLUMNS = (
    COMPANY_COLUMN,
    YEAR_COLUMN,
    TARGET_COLUMN,
    "model",
    "predicted_probability",
    "predicted_class",
)


def create_validation_baseline_predictions(model_data: pd.DataFrame) -> pd.DataFrame:
    """Return majority and training-prevalence predictions for validation rows.

    Parameters
    ----------
    model_data:
        Feature table containing company identifiers, fiscal years, the binary target,
        and pre-assigned ``sample_period`` labels.

    Returns
    -------
    pandas.DataFrame
        Two predictions per validation observation, identified by the ``model`` column.

    Raises
    ------
    ValueError
        If required columns or either the training or validation sample are absent.

    """
    required_columns = {
        COMPANY_COLUMN,
        YEAR_COLUMN,
        TARGET_COLUMN,
        "sample_period",
    }
    missing_columns = required_columns.difference(model_data.columns)
    if missing_columns:
        raise ValueError(f"Required baseline columns are missing: {sorted(missing_columns)}")

    training = model_data.loc[model_data["sample_period"].eq("train")]
    validation = model_data.loc[model_data["sample_period"].eq("validation")]
    if training.empty or validation.empty:
        raise ValueError("Training and validation observations are required for baselines.")
    if not training[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("The training target must contain only zero and one.")

    prevalence = float(training[TARGET_COLUMN].mean())
    identifiers = validation.loc[:, [COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN]].copy()

    predictions: list[pd.DataFrame] = []
    for model_name, probability in (
        ("majority_non_bankrupt", 0.0),
        ("training_prevalence", prevalence),
    ):
        model_predictions = identifiers.copy()
        model_predictions["model"] = model_name
        model_predictions["predicted_probability"] = probability
        model_predictions["predicted_class"] = int(probability >= 0.5)
        predictions.append(model_predictions)

    return pd.concat(predictions, ignore_index=True).loc[:, PREDICTION_COLUMNS]
