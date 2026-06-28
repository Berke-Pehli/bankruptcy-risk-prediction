"""Provide leakage-safe preprocessing for linear and tree classifiers.

Financial ratios can be missing when a denominator is zero and can become extreme when
a denominator is close to zero. The transformers in this module are fitted only on the
training observations supplied by a model pipeline.

Linear-model preprocessing:
    - Median imputation.
    - Column-wise quantile clipping.
    - Standardization to zero mean and unit variance.

Tree-model preprocessing:
    - Median imputation.
    - Column-wise quantile clipping.
    - No scaling, because tree splits are invariant to monotonic rescaling.

No transformation is fitted to the complete dataset before temporal validation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Clip each feature to quantiles estimated from the fitting observations.

    Parameters
    ----------
    lower:
        Lower quantile used as the minimum accepted value for each column.
    upper:
        Upper quantile used as the maximum accepted value for each column.

    """

    def __init__(self, lower: float = 0.005, upper: float = 0.995) -> None:
        """Initialize the lower and upper clipping quantiles."""
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> QuantileClipper:
        """Estimate clipping boundaries from the fitting sample."""
        del y
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Quantile bounds must satisfy 0 <= lower < upper <= 1.")

        values = check_array(X, ensure_all_finite=True, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.lower_bounds_ = np.quantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.quantile(values, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted clipping boundaries to new observations."""
        check_is_fitted(self, attributes=["lower_bounds_", "upper_bounds_"])
        values = check_array(X, ensure_all_finite=True, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("The number of features differs from the fitting sample.")
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return unchanged feature names for pipeline introspection."""
        check_is_fitted(self, attributes=["n_features_in_"])
        if input_features is None:
            return np.asarray(
                [f"x{index}" for index in range(self.n_features_in_)],
                dtype=object,
            )
        if len(input_features) != self.n_features_in_:
            raise ValueError("Input feature names do not match the fitted feature count.")
        return np.asarray(input_features, dtype=object)


def make_linear_preprocessor() -> Pipeline:
    """Create imputation, clipping, and scaling for Logistic Regression."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper()),
            ("scaler", StandardScaler()),
        ]
    )


def make_tree_preprocessor() -> Pipeline:
    """Create imputation and clipping for unscaled tree-based models."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper()),
        ]
    )
