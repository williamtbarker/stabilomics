"""Robust stability selection for scientific feature tables."""

from stabilomics.model import FitResult, SolverError, fit_lad_lasso
from stabilomics.stability import StabilityConfig, StabilityResult, stability_select

__all__ = [
    "FitResult",
    "SolverError",
    "StabilityConfig",
    "StabilityResult",
    "fit_lad_lasso",
    "stability_select",
]

__version__ = "0.1.0"
