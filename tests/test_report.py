from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stabilomics.io import Dataset
from stabilomics.report import (
    build_report,
    evaluate_gates,
    write_features_csv,
    write_report_json,
    write_report_markdown,
)
from stabilomics.stability import StabilityConfig, StabilityResult


def _result() -> StabilityResult:
    selected = np.asarray([[True, False], [True, True], [True, False]])
    coefficients = np.asarray([[1.0, 0.0], [1.2, -0.2], [0.9, 0.0]])
    return StabilityResult(
        selection_frequency=np.asarray([1.0, 1 / 3]),
        selected_consensus=np.asarray([True, False]),
        coefficient_median_selected=np.asarray([1.0, -0.2]),
        coefficient_median_all=np.asarray([1.0, 0.0]),
        sign_consistency=np.asarray([1.0, 1.0]),
        selections_per_fit=np.asarray([1, 2, 1]),
        pairwise_jaccard=np.asarray([0.5, 1.0, 0.5]),
        coefficients=coefficients,
        selected=selected,
        subsample_size=7,
        outcome_center=0.0,
        outcome_scale=1.0,
    )


def test_gates_pass_and_fail() -> None:
    result = _result()
    assert evaluate_gates(result, min_stable_features=1, min_median_jaccard=0.5) == ("PASS", [])
    status, failures = evaluate_gates(result, min_stable_features=2, min_median_jaccard=0.9)
    assert status == "FAIL"
    assert len(failures) == 2


def test_report_writers_are_deterministic(tmp_path: Path) -> None:
    dataset = Dataset(
        outcome=np.arange(10, dtype=float),
        features=np.ones((10, 2)),
        covariates=np.empty((10, 0)),
        feature_names=("gene_a", "gene_b"),
        covariate_names=(),
        row_ids=tuple(str(index) for index in range(10)),
        row_count=10,
    )
    report = build_report(
        dataset,
        _result(),
        StabilityConfig(subsamples=3, fraction=0.7),
        input_name="data.csv",
        input_sha256="a" * 64,
        status="PASS",
        gate_failures=[],
    )
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    csv_path = tmp_path / "features.csv"
    write_report_json(report, json_path)
    write_report_markdown(report, markdown_path)
    write_features_csv(report, csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "PASS"
    assert "`gene_a`" in markdown_path.read_text(encoding="utf-8")
    assert csv_path.read_text(encoding="utf-8").splitlines()[1].startswith("gene_a,1.0")
