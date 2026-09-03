from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from stabilomics.io import load_dataset, sha256_file


def test_load_csv_with_prefix_and_covariate(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text(
        "id,y,age,gene_a,gene_b,ignore\na,1,20,0.1,0.2,9\nb,2,30,0.3,0.4,8\n",
        encoding="utf-8",
    )
    dataset = load_dataset(
        path,
        outcome_column="y",
        covariate_columns=["age"],
        id_column="id",
        feature_prefix="gene_",
    )
    assert dataset.feature_names == ("gene_a", "gene_b")
    assert dataset.covariate_names == ("age",)
    assert dataset.row_ids == ("a", "b")
    assert dataset.features.shape == (2, 2)
    assert len(sha256_file(path)) == 64


def test_load_gzip_by_magic_bytes(tmp_path: Path) -> None:
    path = tmp_path / "odd.tsv"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("id\ty\tf1\nA\t1\t2\nB\t2\t3\n")
    dataset = load_dataset(path, outcome_column="y", id_column="id")
    assert dataset.feature_names == ("f1",)
    assert dataset.outcome.tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "empty"),
        ("y,f\n1,2\n", "at least two"),
        ("y,f,f\n1,2,3\n2,3,4\n", "duplicate header"),
        ("y,f\n1,2\n2\n", "expected 2"),
        ("y,f\n1,2\n2,nope\n", "non-numeric"),
        ("y,f\n1,2\n2,nan\n", "non-finite"),
    ],
)
def test_invalid_tables(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_dataset(path, outcome_column="y")


def test_column_contract_errors(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("id,y,f\na,1,2\nb,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing requested"):
        load_dataset(path, outcome_column="missing")
    with pytest.raises(ValueError, match="no feature"):
        load_dataset(path, outcome_column="y", id_column="id", feature_prefix="gene_")
    with pytest.raises(ValueError, match="unique"):
        load_dataset(path, outcome_column="y", covariate_columns=["f", "f"])
    with pytest.raises(ValueError, match="also be a covariate"):
        load_dataset(path, outcome_column="y", covariate_columns=["y"])


def test_duplicate_row_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("id,y,f\na,1,2\na,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row IDs must be unique"):
        load_dataset(path, outcome_column="y", id_column="id")


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_dataset(tmp_path / "missing.csv", outcome_column="y")
