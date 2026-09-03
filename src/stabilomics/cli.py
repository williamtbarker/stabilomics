"""Command-line interface for reproducible robust stability selection."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from stabilomics import __version__
from stabilomics.io import load_dataset, sha256_file
from stabilomics.report import (
    build_report,
    evaluate_gates,
    write_features_csv,
    write_report_json,
    write_report_markdown,
)
from stabilomics.simulate import simulate_dataset
from stabilomics.stability import StabilityConfig, stability_select


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stabilomics",
        description="Robust stability selection for high-dimensional scientific tables.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="write a deterministic heavy-tailed fixture")
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--samples", type=int, default=160)
    simulate.add_argument("--features", type=int, default=24)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--contamination", type=float, default=0.08)

    fit = subparsers.add_parser("fit", help="run LAD-LASSO stability selection")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--outcome", required=True, help="numeric outcome column")
    fit.add_argument("--id-column", help="optional unique row identifier")
    fit.add_argument(
        "--covariate",
        action="append",
        default=[],
        help="unpenalized numeric covariate; may be repeated",
    )
    fit.add_argument("--feature-prefix", help="only use feature columns with this prefix")
    fit.add_argument("--alpha", type=float, default=0.05)
    fit.add_argument("--subsamples", type=int, default=40)
    fit.add_argument("--fraction", type=float, default=0.7)
    fit.add_argument("--threshold", type=float, default=0.8)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--coefficient-tolerance", type=float, default=1e-8)
    fit.add_argument("--min-stable-features", type=int)
    fit.add_argument("--min-median-jaccard", type=float)
    fit.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run_simulate(arguments: argparse.Namespace) -> int:
    simulate_dataset(
        arguments.output,
        samples=arguments.samples,
        features=arguments.features,
        seed=arguments.seed,
        contamination=arguments.contamination,
    )
    print(f"wrote {arguments.output}")
    return 0


def _run_fit(arguments: argparse.Namespace) -> int:
    dataset = load_dataset(
        arguments.input,
        outcome_column=arguments.outcome,
        covariate_columns=arguments.covariate,
        id_column=arguments.id_column,
        feature_prefix=arguments.feature_prefix,
    )
    config = StabilityConfig(
        alpha=arguments.alpha,
        subsamples=arguments.subsamples,
        fraction=arguments.fraction,
        threshold=arguments.threshold,
        seed=arguments.seed,
        coefficient_tolerance=arguments.coefficient_tolerance,
    )
    result = stability_select(
        dataset.features,
        dataset.outcome,
        covariates=dataset.covariates,
        config=config,
    )
    status, failures = evaluate_gates(
        result,
        min_stable_features=arguments.min_stable_features,
        min_median_jaccard=arguments.min_median_jaccard,
    )
    report = build_report(
        dataset,
        result,
        config,
        input_name=arguments.input.name,
        input_sha256=sha256_file(arguments.input),
        status=status,
        gate_failures=failures,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_report_json(report, arguments.output_dir / "report.json")
    write_report_markdown(report, arguments.output_dir / "report.md")
    write_features_csv(report, arguments.output_dir / "features.csv")
    stable_count = report["stability"]["stable_feature_count"]
    print(
        f"{status}: {stable_count} consensus features; "
        f"median pairwise Jaccard {result.median_pairwise_jaccard:.3f}"
    )
    print(f"wrote {arguments.output_dir}")
    return 2 if status == "FAIL" else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "simulate":
            return _run_simulate(arguments)
        return _run_fit(arguments)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
