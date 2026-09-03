"""Linear-programming solver for LAD-LASSO with unpenalized covariates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog  # type: ignore[import-untyped]

FloatArray = npt.NDArray[np.float64]


class SolverError(RuntimeError):
    """Raised when the linear-programming solver cannot produce a solution."""


@dataclass(frozen=True)
class FitResult:
    """Result of one standardized LAD-LASSO fit."""

    coefficients: FloatArray
    covariate_coefficients: FloatArray
    intercept: float
    objective: float
    selected: npt.NDArray[np.bool_]


def _as_finite_matrix(value: npt.ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_finite_vector(value: npt.ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def fit_lad_lasso(
    features: npt.ArrayLike,
    outcome: npt.ArrayLike,
    *,
    covariates: npt.ArrayLike | None = None,
    alpha: float = 0.05,
    coefficient_tolerance: float = 1e-8,
) -> FitResult:
    """Fit median-regression LASSO with an intercept and optional free covariates.

    The optimized objective is ``mean(abs(y - X beta - C gamma)) +
    alpha * sum(abs(beta))``. The intercept and covariate coefficients are not
    penalized. The problem is represented exactly as a linear program and solved
    with SciPy's HiGHS backend.
    """

    x = _as_finite_matrix(features, "features")
    y = _as_finite_vector(outcome, "outcome")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and outcome must have the same number of rows")
    if x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("at least two rows and one feature are required")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and greater than zero")
    if not np.isfinite(coefficient_tolerance) or coefficient_tolerance < 0:
        raise ValueError("coefficient_tolerance must be finite and non-negative")

    cov: FloatArray
    if covariates is None:
        cov = np.empty((x.shape[0], 0), dtype=np.float64)
    else:
        cov = _as_finite_matrix(covariates, "covariates")
        if cov.shape[0] != x.shape[0]:
            raise ValueError("covariates and features must have the same number of rows")

    n_rows, n_features = x.shape
    nuisance = np.column_stack((np.ones(n_rows, dtype=np.float64), cov))
    n_nuisance = nuisance.shape[1]

    # beta = beta_positive - beta_negative; residual = positive - negative.
    identity = np.eye(n_rows, dtype=np.float64)
    a_eq = np.hstack((x, -x, nuisance, identity, -identity))
    objective = np.concatenate(
        (
            np.full(2 * n_features, alpha, dtype=np.float64),
            np.zeros(n_nuisance, dtype=np.float64),
            np.full(2 * n_rows, 1.0 / n_rows, dtype=np.float64),
        )
    )
    bounds = (
        [(0.0, None)] * (2 * n_features)
        + [(None, None)] * n_nuisance
        + [(0.0, None)] * (2 * n_rows)
    )

    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=y,
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        detail = result.message if result.message else "unknown solver failure"
        raise SolverError(f"LAD-LASSO optimization failed: {detail}")

    beta = result.x[:n_features] - result.x[n_features : 2 * n_features]
    nuisance_start = 2 * n_features
    nuisance_values = result.x[nuisance_start : nuisance_start + n_nuisance]
    beta[np.abs(beta) <= coefficient_tolerance] = 0.0
    return FitResult(
        coefficients=beta,
        covariate_coefficients=nuisance_values[1:].copy(),
        intercept=float(nuisance_values[0]),
        objective=float(result.fun),
        selected=np.abs(beta) > coefficient_tolerance,
    )
