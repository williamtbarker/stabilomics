"""Deterministic heavy-tailed example-data generator."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def simulate_dataset(
    output: Path,
    *,
    samples: int = 160,
    features: int = 24,
    seed: int = 7,
    contamination: float = 0.08,
) -> None:
    """Write a synthetic correlated table with three informative features."""

    if samples < 20:
        raise ValueError("samples must be at least 20")
    if features < 4:
        raise ValueError("features must be at least four")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not 0 <= contamination < 0.5:
        raise ValueError("contamination must be in [0, 0.5)")

    generator = np.random.default_rng(seed)
    indices = np.arange(features)
    covariance = np.power(0.35, np.abs(indices[:, None] - indices[None, :]))
    x = generator.multivariate_normal(np.zeros(features), covariance, size=samples)
    age = generator.normal(50.0, 11.0, size=samples)
    noise = generator.standard_t(df=2.0, size=samples) * 0.35
    outcome = 2.2 * x[:, 0] - 1.7 * x[:, 1] + 1.1 * x[:, 2] + 0.025 * age + noise
    contaminated = max(1, int(np.floor(samples * contamination))) if contamination else 0
    if contaminated:
        rows = generator.choice(samples, size=contaminated, replace=False)
        directions = generator.choice(np.asarray([-1.0, 1.0]), size=contaminated)
        outcome[rows] += directions * generator.uniform(8.0, 14.0, size=contaminated)

    output.parent.mkdir(parents=True, exist_ok=True)
    names = [f"gene_{index + 1:03d}" for index in range(features)]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_id", "phenotype", "age", *names])
        for row_index in range(samples):
            writer.writerow(
                [
                    f"sample_{row_index + 1:03d}",
                    f"{outcome[row_index]:.10f}",
                    f"{age[row_index]:.10f}",
                    *(f"{value:.10f}" for value in x[row_index]),
                ]
            )
