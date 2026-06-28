"""Pytask entry point for validating the canonical raw dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pytask import Product

from bankruptcy_risk.config import (
    CONFIG_MODULE_PATH,
    DATA_VALIDATION_MODULE_PATH,
    RAW_DATA_PATH,
    RAW_VALIDATION_REPORT_PATH,
)
from bankruptcy_risk.data_validation import load_and_validate_raw_data


def task_validate_raw_data(
    raw_data_path: Path = RAW_DATA_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    validation_module_path: Path = DATA_VALIDATION_MODULE_PATH,
    report_path: Annotated[Path, Product] = RAW_VALIDATION_REPORT_PATH,
) -> None:
    """Validate the source CSV and save a machine-readable audit report."""
    if not config_module_path.exists() or not validation_module_path.exists():
        raise FileNotFoundError("Validation source modules must exist before the task runs.")

    _, report = load_and_validate_raw_data(raw_data_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
