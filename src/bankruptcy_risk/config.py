"""Central paths and reproducibility settings for the project.

Keeping shared locations and random seeds in one module prevents analytical scripts
from silently using different files or stochastic settings.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "src" / "bankruptcy_risk"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
METADATA_DIR = OUTPUT_DIR / "metadata"

RAW_DATA_PATH = RAW_DATA_DIR / "american_bankruptcy.csv"
INTERIM_PANEL_PATH = INTERIM_DATA_DIR / "bankruptcy_panel.parquet"
MODEL_FEATURES_PATH = PROCESSED_DATA_DIR / "model_features.parquet"
RAW_VALIDATION_REPORT_PATH = METADATA_DIR / "raw_data_validation.json"
FEATURE_DICTIONARY_PATH = OUTPUT_DIR / "tables" / "feature_dictionary.csv"

CONFIG_MODULE_PATH = PACKAGE_DIR / "config.py"
DATA_VALIDATION_MODULE_PATH = PACKAGE_DIR / "data_validation.py"
TARGET_MODULE_PATH = PACKAGE_DIR / "target.py"
FEATURES_MODULE_PATH = PACKAGE_DIR / "features.py"

RANDOM_SEED = 42
