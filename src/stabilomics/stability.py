"""Repeated-subsampling stability selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from stabilomics.model import fit_lad_lasso
from stabilomics.preprocess import robust_scale, robust_scale_vector

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class StabilityConfig:
    """Parameters controlling deterministic stability selection."""

    alpha: float = 0.05
    subsamples: int = 40
    fraction: float = 0.7
    threshold: float = 0.8
    seed: int = 42
    coefficient_tolerance: float = 1e-8

    def validate(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and greater than zero")
        if self.subsamples < 2:
            raise ValueError("subsamples must be at least two")
        if not np.isfinite(self.fraction) or not 0 < self.fraction < 1:
            raise ValueError("fraction must be strictly between zero and one")
        if not np.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not np.isfinite(self.coefficient_tolerance) or self.coefficient_tolerance < 0:
            raise ValueError("coefficient_tolerance must be finite and non-negative")


@dataclass(frozen=True)
class StabilityResult:
    """Aggregated feature-selection evidence across subsamples."""

    selection_frequency: FloatArray
    selected_consensus: npt.NDArray[np.bool_]
    coefficient_median_selected: FloatArray
    coefficient_median_all: FloatArray
    sign_consistency: FloatArray
    selections_per_fit: npt.NDArray[np.int64]
    pairwise_jaccard: FloatArray
    coefficients: FloatArray
    selected: npt.NDArray[np.bool_]
    subsample_size: int
    outcome_center: float
    outcome_scale: float

    @property
    def median_pairwise_jaccard(self) -> float:
        return float(np.median(self.pairwise_jaccard))


def _pairwise_jaccard(selected: npt.NDArray[np.bool_]) -> FloatArray:
    scores: list[float] = []
    for first, second in combinations(selected, 2):
        union = int(np.count_nonzero(first | second))
        intersection = int(np.count_nonzero(first & second))
        scores.append(1.0 if union == 0 else intersection / union)
    return np.asarray(scores, dtype=np.float64)


def stability_select(
    features: npt.ArrayLike,
    outcome: npt.ArrayLike,
    *,
    covariates: npt.ArrayLike | None = None,
    config: StabilityConfig | None = None,
) -> StabilityResult:
    """Run robust LAD-LASSO on deterministic random subsamples and aggregate evidence."""

    active_config = config if config is not None else StabilityConfig()
    active_config.validate()
    x_scaled, _ = robust_scale(features, name="features")
    y_scaled, y_center, y_scale = robust_scale_vector(outcome, name="outcome")

    if covariates is None:
        cov_scaled: FloatArray | None = None
    else:
        cov_array = np.asarray(covariates, dtype=np.float64)
        if cov_array.shape[1] == 0:
            cov_scaled = None
        else:
            cov_scaled, _ = robust_scale(cov_array, name="covariates")

    n_rows, n_features = x_scaled.shape
    subsample_size = int(np.floor(n_rows * active_config.fraction))
    if subsample_size < 2:
        raise ValueError("fraction leaves fewer than two rows per subsample")

    generator = np.random.default_rng(active_config.seed)
    coefficients = np.zeros((active_config.subsamples, n_features), dtype=np.float64)
    selected = np.zeros((active_config.subsamples, n_features), dtype=np.bool_)
    for index in range(active_config.subsamples):
        rows = np.sort(generator.choice(n_rows, size=subsample_size, replace=False))
        cov_subset = None if cov_scaled is None else cov_scaled[rows]
        fit = fit_lad_lasso(
            x_scaled[rows],
            y_scaled[rows],
            covariates=cov_subset,
            alpha=active_config.alpha,
            coefficient_tolerance=active_config.coefficient_tolerance,
        )
        coefficients[index] = fit.coefficients
        selected[index] = fit.selected

    frequency = np.mean(selected, axis=0)
    selected_consensus = frequency >= active_config.threshold
    median_all = np.median(coefficients, axis=0)
    median_selected = np.zeros(n_features, dtype=np.float64)
    sign_consistency = np.zeros(n_features, dtype=np.float64)
    for feature_index in range(n_features):
        nonzero = coefficients[selected[:, feature_index], feature_index]
        if nonzero.size:
            median_selected[feature_index] = float(np.median(nonzero))
            positive = int(np.count_nonzero(nonzero > 0))
            negative = int(np.count_nonzero(nonzero < 0))
            sign_consistency[feature_index] = max(positive, negative) / nonzero.size

    selections_per_fit = np.asarray(np.sum(selected, axis=1), dtype=np.int64)
    return StabilityResult(
        selection_frequency=frequency,
        selected_consensus=selected_consensus,
        coefficient_median_selected=median_selected,
        coefficient_median_all=median_all,
        sign_consistency=sign_consistency,
        selections_per_fit=selections_per_fit,
        pairwise_jaccard=_pairwise_jaccard(selected),
        coefficients=coefficients,
        selected=selected,
        subsample_size=subsample_size,
        outcome_center=y_center,
        outcome_scale=y_scale,
    )
