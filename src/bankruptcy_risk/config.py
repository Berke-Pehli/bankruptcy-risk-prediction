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
MODEL_DATASET_PATH = PROCESSED_DATA_DIR / "model_dataset.parquet"
RAW_VALIDATION_REPORT_PATH = METADATA_DIR / "raw_data_validation.json"
FEATURE_DICTIONARY_PATH = OUTPUT_DIR / "tables" / "feature_dictionary.csv"
TEMPORAL_SPLIT_SUMMARY_PATH = OUTPUT_DIR / "tables" / "temporal_split_summary.csv"
EXPANDING_FOLD_SUMMARY_PATH = OUTPUT_DIR / "tables" / "expanding_fold_summary.csv"
ANNUAL_OVERVIEW_TABLE_PATH = OUTPUT_DIR / "tables" / "annual_bankruptcy_overview.csv"
TRAINING_RATIO_SUMMARY_PATH = OUTPUT_DIR / "tables" / "training_ratio_summary.csv"

ANNUAL_OVERVIEW_FIGURE_PATH = OUTPUT_DIR / "figures" / "annual_bankruptcy_overview.png"
PERIOD_BALANCE_FIGURE_PATH = OUTPUT_DIR / "figures" / "class_balance_by_period.png"
RATIO_DISTRIBUTION_FIGURE_PATH = (
    OUTPUT_DIR / "figures" / "training_ratio_distributions.png"
)
RATIO_CORRELATION_FIGURE_PATH = OUTPUT_DIR / "figures" / "training_ratio_correlation.png"

CONFIG_MODULE_PATH = PACKAGE_DIR / "config.py"
DATA_VALIDATION_MODULE_PATH = PACKAGE_DIR / "data_validation.py"
TARGET_MODULE_PATH = PACKAGE_DIR / "target.py"
FEATURES_MODULE_PATH = PACKAGE_DIR / "features.py"
SPLITTING_MODULE_PATH = PACKAGE_DIR / "splitting.py"
PREPROCESSING_MODULE_PATH = PACKAGE_DIR / "preprocessing.py"
EXPLORATION_MODULE_PATH = PACKAGE_DIR / "exploration.py"
VISUALIZATION_MODULE_PATH = PACKAGE_DIR / "visualization.py"

RANDOM_SEED = 42
