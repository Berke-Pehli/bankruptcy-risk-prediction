from __future__ import annotations

import numpy as np
import pytest

from bankruptcy_risk.preprocessing import (
    QuantileClipper,
    make_linear_preprocessor,
    make_tree_preprocessor,
)


def test_quantile_clipper_limits_values_using_fitted_sample() -> None:
    training = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    future = np.array([[-1_000.0], [1_000.0]])
    clipper = QuantileClipper(lower=0.2, upper=0.8).fit(training)

    transformed = clipper.transform(future)

    assert transformed[0, 0] == pytest.approx(np.quantile(training[:, 0], 0.2))
    assert transformed[1, 0] == pytest.approx(np.quantile(training[:, 0], 0.8))


def test_quantile_clipper_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="Quantile bounds"):
        QuantileClipper(lower=0.9, upper=0.1).fit(np.ones((3, 2)))


def test_linear_preprocessor_imputes_clips_and_scales() -> None:
    training = np.array(
        [
            [1.0, np.nan],
            [2.0, 2.0],
            [3.0, 4.0],
            [100.0, 6.0],
        ]
    )

    transformed = make_linear_preprocessor().fit_transform(training)

    assert transformed.shape == training.shape
    assert np.isfinite(transformed).all()
    assert transformed.mean(axis=0) == pytest.approx(np.zeros(2), abs=1e-12)


def test_tree_preprocessor_does_not_standardize_values() -> None:
    training = np.array([[1.0], [2.0], [3.0], [4.0]])

    transformed = make_tree_preprocessor().fit_transform(training)

    assert transformed.shape == training.shape
    assert transformed.mean() != pytest.approx(0.0)


def test_feature_names_pass_through_quantile_clipper() -> None:
    clipper = QuantileClipper().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    names = clipper.get_feature_names_out(["profitability", "leverage"])

    assert names.tolist() == ["profitability", "leverage"]

