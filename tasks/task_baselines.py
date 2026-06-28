"""Generate simple validation-period bankruptcy benchmarks.

Inputs:
    - ``data/processed/model_dataset.parquet``
    - Baseline and configuration source modules tracked by Pytask.

Output:
    - ``outputs/tables/validation_baseline_predictions.csv``

The task estimates the bankruptcy prevalence from the training years only. It never
uses validation outcomes to construct either reference prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from bankruptcy_risk.baselines import create_validation_baseline_predictions
from bankruptcy_risk.config import (
    BASELINES_MODULE_PATH,
    CONFIG_MODULE_PATH,
    MODEL_DATASET_PATH,
    VALIDATION_BASELINE_PREDICTIONS_PATH,
)


def task_create_validation_baselines(
    model_data_path: Path = MODEL_DATASET_PATH,
    config_module_path: Path = CONFIG_MODULE_PATH,
    baselines_module_path: Path = BASELINES_MODULE_PATH,
    predictions_path: Annotated[Path, Product] = VALIDATION_BASELINE_PREDICTIONS_PATH,
) -> None:
    """Write majority-class and training-prevalence validation predictions."""
    if not config_module_path.exists() or not baselines_module_path.exists():
        raise FileNotFoundError("Baseline source modules must exist.")

    model_data = pd.read_parquet(model_data_path)
    predictions = create_validation_baseline_predictions(model_data)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
