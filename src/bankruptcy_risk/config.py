"""Central paths and reproducibility settings for the project.

Keeping shared locations and random seeds in one module prevents analytical scripts
from silently using different files or stochastic settings.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
METADATA_DIR = OUTPUT_DIR / "metadata"

RAW_DATA_PATH = RAW_DATA_DIR / "american_bankruptcy.csv"
RAW_VALIDATION_REPORT_PATH = METADATA_DIR / "raw_data_validation.json"

RANDOM_SEED = 42

