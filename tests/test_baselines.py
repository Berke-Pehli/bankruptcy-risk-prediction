from __future__ import annotations

import pandas as pd
import pytest

from bankruptcy_risk.baselines import create_validation_baseline_predictions
from bankruptcy_risk.target import TARGET_COLUMN, YEAR_COLUMN


@pytest.fixture
def model_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_id": ["A", "B", "C", "D", "E"],
            YEAR_COLUMN: [2010, 2011, 2011, 2012, 2012],
            TARGET_COLUMN: [0, 0, 1, 0, 1],
            "sample_period": ["train", "train", "train", "validation", "validation"],
        }
    )


def test_baselines_create_two_predictions_per_validation_row(
    model_data: pd.DataFrame,
) -> None:
    predictions = create_validation_baseline_predictions(model_data)

    assert len(predictions) == 4
    assert predictions["model"].value_counts().to_dict() == {
        "majority_non_bankrupt": 2,
        "training_prevalence": 2,
    }
    assert set(predictions[YEAR_COLUMN]) == {2012}


def test_prevalence_baseline_uses_training_outcomes_only(model_data: pd.DataFrame) -> None:
    predictions = create_validation_baseline_predictions(model_data)
    prevalence_predictions = predictions.loc[
        predictions["model"].eq("training_prevalence"), "predicted_probability"
    ]

    assert prevalence_predictions.eq(1 / 3).all()


def test_majority_baseline_always_predicts_non_bankruptcy(model_data: pd.DataFrame) -> None:
    predictions = create_validation_baseline_predictions(model_data)
    majority = predictions.loc[predictions["model"].eq("majority_non_bankrupt")]

    assert majority["predicted_probability"].eq(0).all()
    assert majority["predicted_class"].eq(0).all()
