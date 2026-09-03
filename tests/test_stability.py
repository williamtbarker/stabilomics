from __future__ import annotations

import numpy as np
import pytest

from stabilomics.stability import StabilityConfig, stability_select


def _signal_data() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(11)
    x = generator.normal(size=(90, 8))
    y = 2.5 * x[:, 0] - 1.8 * x[:, 1] + generator.standard_t(2, size=90) * 0.15
    return x, y


def test_stability_selection_is_deterministic_and_finds_signals() -> None:
    x, y = _signal_data()
    config = StabilityConfig(alpha=0.08, subsamples=16, fraction=0.7, threshold=0.8, seed=9)
    first = stability_select(x, y, config=config)
    second = stability_select(x, y, config=config)
    assert np.array_equal(first.selected, second.selected)
    assert np.array_equal(first.coefficients, second.coefficients)
    assert first.selected_consensus[:2].tolist() == [True, True]
    assert first.sign_consistency[0] == pytest.approx(1.0)
    assert first.sign_consistency[1] == pytest.approx(1.0)
    assert first.subsample_size == 62
    assert 0 <= first.median_pairwise_jaccard <= 1


def test_empty_selection_sets_have_jaccard_one() -> None:
    x, y = _signal_data()
    result = stability_select(
        x,
        y,
        config=StabilityConfig(alpha=100.0, subsamples=3, fraction=0.6, threshold=1.0),
    )
    assert not result.selected.any()
    assert result.pairwise_jaccard.tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "config",
    [
        StabilityConfig(alpha=0),
        StabilityConfig(subsamples=1),
        StabilityConfig(fraction=0),
        StabilityConfig(fraction=1),
        StabilityConfig(threshold=0),
        StabilityConfig(threshold=1.1),
        StabilityConfig(seed=-1),
        StabilityConfig(coefficient_tolerance=-1),
    ],
)
def test_invalid_configuration_is_rejected(config: StabilityConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_fraction_cannot_leave_too_few_rows() -> None:
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    with pytest.raises(ValueError, match="fewer than two"):
        stability_select(x, y, config=StabilityConfig(fraction=0.2, subsamples=2))


def test_covariate_path_runs() -> None:
    x, y = _signal_data()
    covariate = np.linspace(0.0, 1.0, x.shape[0])[:, None]
    result = stability_select(
        x,
        y + covariate[:, 0],
        covariates=covariate,
        config=StabilityConfig(alpha=0.1, subsamples=4, fraction=0.7),
    )
    assert result.coefficients.shape == (4, 8)
