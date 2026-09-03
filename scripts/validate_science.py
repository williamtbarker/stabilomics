"""Controlled validation of robust selection on paired synthetic datasets."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from stabilomics.io import load_dataset
from stabilomics.simulate import simulate_dataset
from stabilomics.stability import StabilityConfig, StabilityResult, stability_select


def _run(path: Path) -> tuple[tuple[str, ...], StabilityResult]:
    dataset = load_dataset(
        path,
        outcome_column="phenotype",
        covariate_columns=["age"],
        id_column="sample_id",
        feature_prefix="gene_",
    )
    result = stability_select(
        dataset.features,
        dataset.outcome,
        covariates=dataset.covariates,
        config=StabilityConfig(
            alpha=0.15,
            subsamples=40,
            fraction=0.7,
            threshold=0.8,
            seed=42,
        ),
    )
    selected = tuple(
        name
        for name, include in zip(dataset.feature_names, result.selected_consensus, strict=True)
        if include
    )
    return selected, result


def main() -> int:
    """Require planted-signal recovery before and after controlled contamination."""

    expected = ("gene_001", "gene_002", "gene_003")
    with TemporaryDirectory(prefix="stabilomics-validation-") as temporary:
        directory = Path(temporary)
        clean_path = directory / "clean.csv"
        contaminated_path = directory / "contaminated.csv"
        simulate_dataset(clean_path, contamination=0.0)
        simulate_dataset(contaminated_path, contamination=0.08)
        clean, clean_result = _run(clean_path)
        contaminated, contaminated_result = _run(contaminated_path)

    if clean != expected:
        raise AssertionError(f"clean consensus mismatch: {clean!r}")
    if contaminated != expected:
        raise AssertionError(f"contaminated consensus mismatch: {contaminated!r}")
    signal_floor = float(np.min(contaminated_result.selection_frequency[:3]))
    if signal_floor < 0.9:
        raise AssertionError(f"contaminated signal-frequency floor is {signal_floor:.3f}")

    print(f"clean consensus: {', '.join(clean)}")
    print(f"contaminated consensus: {', '.join(contaminated)}")
    print(f"contaminated signal-frequency floor: {signal_floor:.3f}")
    print(f"clean median pairwise Jaccard: {clean_result.median_pairwise_jaccard:.3f}")
    print(
        f"contaminated median pairwise Jaccard: {contaminated_result.median_pairwise_jaccard:.3f}"
    )
    print("scientific fixture validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
