"""Deterministic machine-readable and human-readable result rendering."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stabilomics.io import Dataset
from stabilomics.stability import StabilityConfig, StabilityResult


def _round(value: float | np.floating[Any]) -> float:
    return round(float(value), 8)


def evaluate_gates(
    result: StabilityResult,
    *,
    min_stable_features: int | None,
    min_median_jaccard: float | None,
) -> tuple[str, list[str]]:
    """Evaluate optional workflow gates without implying biological validity."""

    failures: list[str] = []
    stable_count = int(np.count_nonzero(result.selected_consensus))
    if min_stable_features is not None:
        if min_stable_features < 0:
            raise ValueError("min_stable_features must be non-negative")
        if stable_count < min_stable_features:
            failures.append(
                f"stable feature count {stable_count} is below required {min_stable_features}"
            )
    if min_median_jaccard is not None:
        if not 0 <= min_median_jaccard <= 1:
            raise ValueError("min_median_jaccard must be in [0, 1]")
        if result.median_pairwise_jaccard < min_median_jaccard:
            failures.append(
                "median pairwise Jaccard "
                f"{result.median_pairwise_jaccard:.4f} is below required "
                f"{min_median_jaccard:.4f}"
            )
    return ("FAIL" if failures else "PASS"), failures


def build_report(
    dataset: Dataset,
    result: StabilityResult,
    config: StabilityConfig,
    *,
    input_name: str,
    input_sha256: str,
    status: str,
    gate_failures: list[str],
) -> dict[str, Any]:
    """Build a stable JSON-compatible report object."""

    order = sorted(
        range(len(dataset.feature_names)),
        key=lambda index: (-result.selection_frequency[index], dataset.feature_names[index]),
    )
    feature_rows = [
        {
            "feature": dataset.feature_names[index],
            "selection_frequency": _round(result.selection_frequency[index]),
            "selected_consensus": bool(result.selected_consensus[index]),
            "coefficient_median_selected": _round(result.coefficient_median_selected[index]),
            "coefficient_median_all": _round(result.coefficient_median_all[index]),
            "sign_consistency": _round(result.sign_consistency[index]),
        }
        for index in order
    ]
    selection_counts = result.selections_per_fit.astype(np.float64)
    jaccard = result.pairwise_jaccard
    return {
        "schema_version": "1.0",
        "tool": {"name": "stabilomics", "version": "0.1.0"},
        "status": status,
        "gate_failures": gate_failures,
        "input": {
            "file": input_name,
            "sha256": input_sha256,
            "rows": dataset.row_count,
            "features": len(dataset.feature_names),
            "covariates": list(dataset.covariate_names),
        },
        "parameters": {
            "alpha": config.alpha,
            "subsamples": config.subsamples,
            "fraction": config.fraction,
            "threshold": config.threshold,
            "seed": config.seed,
            "coefficient_tolerance": config.coefficient_tolerance,
            "subsample_size": result.subsample_size,
        },
        "stability": {
            "stable_feature_count": int(np.count_nonzero(result.selected_consensus)),
            "selected_per_fit_min": int(np.min(selection_counts)),
            "selected_per_fit_median": _round(np.median(selection_counts)),
            "selected_per_fit_max": int(np.max(selection_counts)),
            "pairwise_jaccard_min": _round(np.min(jaccard)),
            "pairwise_jaccard_median": _round(np.median(jaccard)),
            "pairwise_jaccard_max": _round(np.max(jaccard)),
        },
        "features": feature_rows,
        "method": {
            "fit": "LAD-LASSO linear program on robustly standardized data",
            "aggregation": "selection frequency across seeded subsamples without replacement",
            "jaccard_empty_set_convention": 1.0,
            "coefficient_scale": "robustly standardized outcome and features",
        },
        "limitations": [
            "Selection stability is not evidence of causality or biological validity.",
            "Correlated predictors can exchange selection across subsamples.",
            "Alpha and the consensus threshold require domain-specific sensitivity analysis.",
            (
                "Pairwise subsample results are dependent and are descriptive, "
                "not confidence intervals."
            ),
        ],
    }


def write_report_json(report: dict[str, Any], output: Path) -> None:
    """Write canonical, diff-friendly JSON."""

    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_features_csv(report: dict[str, Any], output: Path) -> None:
    """Write the ranked feature evidence table."""

    fields = [
        "feature",
        "selection_frequency",
        "selected_consensus",
        "coefficient_median_selected",
        "coefficient_median_all",
        "sign_consistency",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["features"])


def write_report_markdown(report: dict[str, Any], output: Path) -> None:
    """Write a concise review report with the most stable features."""

    parameters = report["parameters"]
    stability = report["stability"]
    selected = [row for row in report["features"] if row["selected_consensus"]]
    lines = [
        "# Stabilomics report",
        "",
        f"**Status:** {report['status']}",
        "",
        (
            f"Analyzed {report['input']['rows']} rows and {report['input']['features']} features "
            f"using {parameters['subsamples']} seeded subsamples of "
            f"{parameters['subsample_size']} rows."
        ),
        "",
        "## Stability summary",
        "",
        f"- Consensus features: {stability['stable_feature_count']}",
        f"- Consensus threshold: {parameters['threshold']:.3f}",
        f"- LAD-LASSO alpha: {parameters['alpha']:.6g}",
        (
            "- Selected features per fit (min/median/max): "
            f"{stability['selected_per_fit_min']}/"
            f"{stability['selected_per_fit_median']:.1f}/"
            f"{stability['selected_per_fit_max']}"
        ),
        (
            "- Pairwise selection-set Jaccard (min/median/max): "
            f"{stability['pairwise_jaccard_min']:.3f}/"
            f"{stability['pairwise_jaccard_median']:.3f}/"
            f"{stability['pairwise_jaccard_max']:.3f}"
        ),
        "",
    ]
    if report["gate_failures"]:
        lines.extend(["## Gate failures", ""])
        lines.extend(f"- {failure}" for failure in report["gate_failures"])
        lines.append("")
    lines.extend(
        [
            "## Consensus features",
            "",
            "| Feature | Frequency | Median selected coefficient | Sign consistency |",
            "|---|---:|---:|---:|",
        ]
    )
    if selected:
        lines.extend(
            (
                f"| `{row['feature']}` | {row['selection_frequency']:.3f} | "
                f"{row['coefficient_median_selected']:.4f} | "
                f"{row['sign_consistency']:.3f} |"
            )
            for row in selected
        )
    else:
        lines.append("| *(none)* | — | — | — |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            *[f"- {limitation}" for limitation in report["limitations"]],
            "",
            "See `features.csv` for every feature and `report.json` for the full audit record.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
