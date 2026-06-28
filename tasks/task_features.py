"""Build model features and their data dictionary.

Inputs:
    - ``data/interim/bankruptcy_panel.parquet``
    - Feature-engineering source modules tracked by Pytask.

Outputs:
    - ``data/processed/model_features.parquet``
    - ``outputs/tables/feature_dictionary.csv``

The task preserves missing values created by undefined ratios. Imputation and clipping
remain part of the later model preprocessing pipeline to avoid look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from bankruptcy_risk.config import (
    CONFIG_MODULE_PATH,
    FEATURE_DICTIONARY_PATH,
    FEATURES_MODULE_PATH,
    INTERIM_PANEL_PATH,
    MODEL_FEATURES_PATH,
    TARGET_MODULE_PATH,
)
from bankruptcy_risk.features import build_feature_dictionary, engineer_model_features


def task_build_model_features(
    panel_path: Path = INTERIM_PANEL_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    target_module_path: Path = TARGET_MODULE_PATH,
    features_module_path: Path = FEATURES_MODULE_PATH,
    feature_path: Annotated[Path, Product] = MODEL_FEATURES_PATH,
) -> None:
    """Create and store the complete firm-year model feature table."""
    source_paths = (config_module_path, target_module_path, features_module_path)
    if not all(path.exists() for path in source_paths):
        raise FileNotFoundError("Feature-engineering source modules must exist.")

    panel = pd.read_parquet(panel_path)
    features = engineer_model_features(panel)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)


def task_write_feature_dictionary(
    features_module_path: Path = FEATURES_MODULE_PATH,
    dictionary_path: Annotated[Path, Product] = FEATURE_DICTIONARY_PATH,
) -> None:
    """Write a human-readable definition and formula for every model feature."""
    if not features_module_path.exists():
        raise FileNotFoundError("The feature-engineering module must exist.")

    dictionary = build_feature_dictionary()
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(dictionary_path, index=False)

