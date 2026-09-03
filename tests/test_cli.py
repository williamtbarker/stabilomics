from __future__ import annotations

import json
from pathlib import Path

import pytest

from stabilomics.cli import main


def test_end_to_end_cli(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.csv"
    assert main(["simulate", "--output", str(fixture), "--samples", "60"]) == 0
    output = tmp_path / "results"
    exit_code = main(
        [
            "fit",
            "--input",
            str(fixture),
            "--outcome",
            "phenotype",
            "--id-column",
            "sample_id",
            "--covariate",
            "age",
            "--feature-prefix",
            "gene_",
            "--alpha",
            "0.06",
            "--subsamples",
            "6",
            "--threshold",
            "0.75",
            "--output-dir",
            str(output),
        ]
    )
    assert exit_code == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["input"]["rows"] == 60
    assert (output / "report.md").is_file()
    assert (output / "features.csv").is_file()


def test_simulate_writes_portable_lf_line_endings(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.csv"
    assert main(["simulate", "--output", str(fixture), "--samples", "20"]) == 0
    contents = fixture.read_bytes()
    assert b"\r\n" not in contents
    assert contents.count(b"\n") == 21


def test_cli_gate_failure_returns_two(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.csv"
    main(["simulate", "--output", str(fixture), "--samples", "40"])
    exit_code = main(
        [
            "fit",
            "--input",
            str(fixture),
            "--outcome",
            "phenotype",
            "--feature-prefix",
            "gene_",
            "--subsamples",
            "3",
            "--min-stable-features",
            "999",
            "--output-dir",
            str(tmp_path / "failed"),
        ]
    )
    assert exit_code == 2


def test_cli_reports_input_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "fit",
                "--input",
                str(tmp_path / "missing.csv"),
                "--outcome",
                "y",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
    assert "input file does not exist" in capsys.readouterr().err
