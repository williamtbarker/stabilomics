from __future__ import annotations

import numpy as np
import pytest

from stabilomics.model import fit_lad_lasso
from stabilomics.preprocess import robust_scale, robust_scale_vector


def test_lad_lasso_recovers_dominant_signal() -> None:
    x = np.linspace(-2.0, 2.0, 21)[:, None]
    y = 1.5 + 2.25 * x[:, 0]
    result = fit_lad_lasso(x, y, alpha=0.001)
    assert result.selected.tolist() == [True]
    assert result.intercept == pytest.approx(1.5, abs=1e-7)
    assert result.coefficients[0] == pytest.approx(2.25, abs=1e-7)
    assert result.objective >= 0


def test_covariate_is_unpenalized() -> None:
    covariate = np.linspace(-3.0, 3.0, 25)
    feature = np.tile(np.asarray([-1.0, 1.0, 0.5, -0.5, 0.25]), 5)[:, None]
    y = 4.0 + 3.0 * covariate
    result = fit_lad_lasso(feature, y, covariates=covariate[:, None], alpha=0.5)
    assert result.selected.tolist() == [False]
    assert result.covariate_coefficients[0] == pytest.approx(3.0, abs=1e-7)


@pytest.mark.parametrize(
    ("features", "outcome", "message"),
    [
        (np.ones(4), np.ones(4), "two-dimensional"),
        (np.ones((4, 1)), np.ones((4, 1)), "one-dimensional"),
        (np.ones((4, 1)), np.ones(3), "same number"),
        (np.ones((1, 1)), np.ones(1), "at least two"),
        (np.asarray([[1.0], [np.nan]]), np.ones(2), "finite"),
    ],
)
def test_lad_lasso_rejects_invalid_shapes(
    features: np.ndarray, outcome: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_lad_lasso(features, outcome)


def test_lad_lasso_rejects_invalid_parameters() -> None:
    x = np.arange(6, dtype=float).reshape(3, 2)
    y = np.arange(3, dtype=float)
    with pytest.raises(ValueError, match="alpha"):
        fit_lad_lasso(x, y, alpha=0)
    with pytest.raises(ValueError, match="coefficient_tolerance"):
        fit_lad_lasso(x, y, coefficient_tolerance=-1)
    with pytest.raises(ValueError, match="covariates and features"):
        fit_lad_lasso(x, y, covariates=np.ones((2, 1)))


def test_robust_scaling() -> None:
    matrix = np.asarray([[1.0, 10.0], [2.0, 12.0], [3.0, 14.0], [100.0, 16.0]])
    scaled, parameters = robust_scale(matrix, name="matrix")
    assert np.median(scaled, axis=0) == pytest.approx([0.0, 0.0])
    assert parameters.center == pytest.approx([2.5, 13.0])
    vector, center, scale = robust_scale_vector(matrix[:, 0], name="vector")
    assert np.median(vector) == pytest.approx(0.0)
    assert center == pytest.approx(2.5)
    assert scale > 0


def test_robust_scaling_rejects_constant_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="constant"):
        robust_scale(np.ones((3, 1)), name="features")
    with pytest.raises(ValueError, match="constant"):
        robust_scale_vector(np.ones(3), name="outcome")
    with pytest.raises(ValueError, match="finite"):
        robust_scale(np.asarray([[1.0], [np.inf]]), name="features")
