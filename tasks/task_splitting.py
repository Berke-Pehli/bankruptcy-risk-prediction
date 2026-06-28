"""Create and document the chronological modelling samples.

Inputs:
    - ``data/processed/model_features.parquet``
    - Splitting and configuration source modules tracked by Pytask.

Outputs:
    - ``data/processed/model_dataset.parquet`` with train, validation, and test labels.
    - Main temporal-split and expanding-window summary tables.

The task only assigns sample membership. It does not fit imputation, clipping, scaling,
or any model to the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from bankruptcy_risk.config import (
    CONFIG_MODULE_PATH,
    EXPANDING_FOLD_SUMMARY_PATH,
    FEATURES_MODULE_PATH,
    MODEL_DATASET_PATH,
    MODEL_FEATURES_PATH,
    SPLITTING_MODULE_PATH,
    TEMPORAL_SPLIT_SUMMARY_PATH,
)
from bankruptcy_risk.splitting import (
    assign_sample_periods,
    summarize_expanding_folds,
    summarize_sample_periods,
)


def task_assign_temporal_periods(
    feature_path: Path = MODEL_FEATURES_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    features_module_path: Path = FEATURES_MODULE_PATH,
    splitting_module_path: Path = SPLITTING_MODULE_PATH,
    model_data_path: Annotated[Path, Product] = MODEL_DATASET_PATH,
) -> None:
    """Assign every firm-year observation to its pre-specified sample period."""
    source_paths = (config_module_path, features_module_path, splitting_module_path)
    if not all(path.exists() for path in source_paths):
        raise FileNotFoundError("Temporal-splitting source modules must exist.")

    features = pd.read_parquet(feature_path)
    model_data = assign_sample_periods(features)
    model_data_path.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_parquet(model_data_path, index=False)


def task_write_temporal_split_summary(
    model_data_path: Path = MODEL_DATASET_PATH,
    splitting_module_path: Path = SPLITTING_MODULE_PATH,
    summary_path: Annotated[Path, Product] = TEMPORAL_SPLIT_SUMMARY_PATH,
) -> None:
    """Write an audit table for the main train, validation, and test periods."""
    if not splitting_module_path.exists():
        raise FileNotFoundError("The temporal-splitting module must exist.")

    model_data = pd.read_parquet(model_data_path)
    summary = summarize_sample_periods(model_data)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)


def task_write_expanding_fold_summary(
    model_data_path: Path = MODEL_DATASET_PATH,
    splitting_module_path: Path = SPLITTING_MODULE_PATH,
    summary_path: Annotated[Path, Product] = EXPANDING_FOLD_SUMMARY_PATH,
) -> None:
    """Write the year boundaries, observations, and events for tuning folds."""
    if not splitting_module_path.exists():
        raise FileNotFoundError("The temporal-splitting module must exist.")

    model_data = pd.read_parquet(model_data_path)
    summary = summarize_expanding_folds(model_data)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

