"""Pytask entry point for constructing the bankruptcy firm-year panel."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pytask import Product

from bankruptcy_risk.config import (
    CONFIG_MODULE_PATH,
    DATA_VALIDATION_MODULE_PATH,
    INTERIM_PANEL_PATH,
    RAW_DATA_PATH,
    RAW_VALIDATION_REPORT_PATH,
    TARGET_MODULE_PATH,
)
from bankruptcy_risk.data_validation import load_and_validate_raw_data
from bankruptcy_risk.target import construct_bankruptcy_panel


def task_construct_bankruptcy_panel(
    raw_data_path: Path = RAW_DATA_PATH,
    validation_report_path: Path = RAW_VALIDATION_REPORT_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    validation_module_path: Path = DATA_VALIDATION_MODULE_PATH,
    target_module_path: Path = TARGET_MODULE_PATH,
    panel_path: Annotated[Path, Product] = INTERIM_PANEL_PATH,
) -> None:
    """Construct and store the audited one-year-ahead prediction panel."""
    if not validation_report_path.exists():
        raise FileNotFoundError("The raw-data validation report must be created first.")
    source_paths = (config_module_path, validation_module_path, target_module_path)
    if not all(path.exists() for path in source_paths):
        raise FileNotFoundError(
            "Panel-construction source modules must exist before the task runs."
        )

    raw_data, _ = load_and_validate_raw_data(raw_data_path)
    panel = construct_bankruptcy_panel(raw_data)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
