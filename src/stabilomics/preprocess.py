"""Robust scaling used before repeated model fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RobustScale:
    """Column-wise medians and interquartile ranges."""

    center: FloatArray
    scale: FloatArray


def robust_scale(matrix: npt.ArrayLike, *, name: str) -> tuple[FloatArray, RobustScale]:
    """Median-center and IQR-scale a matrix, rejecting constant columns."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = np.asarray(q75 - q25, dtype=np.float64)
    constant = scale <= np.finfo(np.float64).eps
    if np.any(constant):
        indices = ", ".join(str(index) for index in np.flatnonzero(constant))
        raise ValueError(f"{name} contains constant or near-constant columns at indices: {indices}")
    return (values - center) / scale, RobustScale(center=center, scale=scale)


def robust_scale_vector(vector: npt.ArrayLike, *, name: str) -> tuple[FloatArray, float, float]:
    """Median-center and IQR-scale a vector."""

    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    center = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} is constant or near-constant")
    return (values - center) / scale, center, scale
