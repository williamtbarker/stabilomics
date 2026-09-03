"""Strict delimited-table input and deterministic file helpers."""

from __future__ import annotations

import csv
import gzip
import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Dataset:
    """Numeric outcome, optional covariates, and named feature matrix."""

    outcome: FloatArray
    features: FloatArray
    covariates: FloatArray
    feature_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    row_ids: tuple[str, ...]
    row_count: int


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delimiter(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return "\t" if suffixes and suffixes[-1] in {".tab", ".tsv"} else ","


@contextmanager
def _open_text(path: Path) -> Iterator[IO[str]]:
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield handle


def _numeric_column(
    rows: Sequence[Sequence[str]],
    index: int,
    name: str,
) -> FloatArray:
    values = np.empty(len(rows), dtype=np.float64)
    for row_index, row in enumerate(rows, start=2):
        raw = row[index].strip()
        if not raw:
            raise ValueError(f"missing value in column {name!r} at row {row_index}")
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(
                f"non-numeric value {raw!r} in column {name!r} at row {row_index}"
            ) from error
        if not np.isfinite(value):
            raise ValueError(f"non-finite value in column {name!r} at row {row_index}")
        values[row_index - 2] = value
    return values


def load_dataset(
    path: Path,
    *,
    outcome_column: str,
    covariate_columns: Sequence[str] = (),
    id_column: str | None = None,
    feature_prefix: str | None = None,
) -> Dataset:
    """Load a strict CSV/TSV table; gzip is detected by content."""

    if not path.is_file():
        raise ValueError(f"input file does not exist: {path}")
    with _open_text(path) as handle:
        reader = csv.reader(handle, delimiter=_delimiter(path))
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("input table is empty") from error
        header = [name.strip() for name in header]
        if not header or any(not name for name in header):
            raise ValueError("header names must be non-empty")
        duplicate_headers = sorted({name for name in header if header.count(name) > 1})
        if duplicate_headers:
            raise ValueError(f"duplicate header names: {', '.join(duplicate_headers)}")
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    if len(rows) < 2:
        raise ValueError("input table must contain at least two data rows")
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f"row {row_number} has {len(row)} columns; expected {len(header)}")

    requested = [outcome_column, *covariate_columns]
    if id_column is not None:
        requested.append(id_column)
    missing = [name for name in requested if name not in header]
    if missing:
        raise ValueError(f"missing requested columns: {', '.join(missing)}")
    if len(set(covariate_columns)) != len(covariate_columns):
        raise ValueError("covariate columns must be unique")
    if outcome_column in covariate_columns:
        raise ValueError("outcome column cannot also be a covariate")

    excluded = {outcome_column, *covariate_columns}
    if id_column is not None:
        excluded.add(id_column)
    feature_names = tuple(
        name
        for name in header
        if name not in excluded and (feature_prefix is None or name.startswith(feature_prefix))
    )
    if not feature_names:
        qualifier = "" if feature_prefix is None else f" matching prefix {feature_prefix!r}"
        raise ValueError(f"no feature columns remain{qualifier}")

    indices = {name: index for index, name in enumerate(header)}
    outcome = _numeric_column(rows, indices[outcome_column], outcome_column)
    features = np.column_stack(
        [_numeric_column(rows, indices[name], name) for name in feature_names]
    )
    if covariate_columns:
        covariates = np.column_stack(
            [_numeric_column(rows, indices[name], name) for name in covariate_columns]
        )
    else:
        covariates = np.empty((len(rows), 0), dtype=np.float64)

    if id_column is None:
        row_ids = tuple(str(index) for index in range(1, len(rows) + 1))
    else:
        row_ids = tuple(row[indices[id_column]].strip() for row in rows)
        if any(not value for value in row_ids):
            raise ValueError("row IDs must be non-empty")
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("row IDs must be unique")

    return Dataset(
        outcome=outcome,
        features=np.asarray(features, dtype=np.float64),
        covariates=np.asarray(covariates, dtype=np.float64),
        feature_names=feature_names,
        covariate_names=tuple(covariate_columns),
        row_ids=row_ids,
        row_count=len(rows),
    )
