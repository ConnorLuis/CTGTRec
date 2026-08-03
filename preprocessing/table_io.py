#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict comma/TSV table readers used by CTGTRec preprocessing.

New mapping files are standard comma-separated CSV. Earlier notebooks and
released artifacts may contain tab-separated data despite the ``.csv`` suffix.
This module detects either format from the header and rejects ambiguous or
unsupported layouts instead of silently reading the entire header as one
column.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


SUPPORTED_DELIMITERS = (",", "\t")


def delimiter_name(delimiter: str) -> str:
    if delimiter == ",":
        return "comma"
    if delimiter == "\t":
        return "tab"
    return repr(delimiter)


def _first_nonempty_line(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Table file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.strip():
                return line.rstrip("\r\n")
    raise ValueError(f"{path}: table file is empty.")


def _parse_header(line: str, delimiter: str) -> list[str]:
    values = next(csv.reader([line], delimiter=delimiter))
    return [str(value).strip() for value in values]


def detect_delimiter(
    path: Path,
    *,
    required_columns: Sequence[str],
    candidates: Sequence[str] = SUPPORTED_DELIMITERS,
) -> str:
    """Detect a supported delimiter by matching all required header columns."""
    if not required_columns:
        raise ValueError("required_columns must not be empty.")

    line = _first_nonempty_line(path)
    required = set(required_columns)
    matches: list[tuple[str, list[str]]] = []
    observed: dict[str, list[str]] = {}

    for delimiter in candidates:
        header = _parse_header(line, delimiter)
        observed[delimiter_name(delimiter)] = header
        if required.issubset(header):
            matches.append((delimiter, header))

    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        names = [delimiter_name(delimiter) for delimiter, _ in matches]
        raise ValueError(
            f"{path}: ambiguous delimiter; required columns "
            f"{sorted(required)} match {names}."
        )

    raise ValueError(
        f"{path}: could not detect comma or tab delimiter containing required "
        f"columns {sorted(required)}. Parsed headers: {observed}."
    )


def read_delimited_table(
    path: Path,
    *,
    required_columns: Sequence[str],
    dtype: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Read comma CSV or legacy TSV and return ``(dataframe, delimiter)``."""
    delimiter = detect_delimiter(
        path,
        required_columns=required_columns,
    )
    table = pd.read_csv(
        path,
        sep=delimiter,
        dtype=dtype,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    normalized_columns = [str(column).strip() for column in table.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError(
            f"{path}: duplicate columns after whitespace normalization: "
            f"{normalized_columns}."
        )
    table.columns = normalized_columns

    missing = [
        column for column in required_columns if column not in table.columns
    ]
    if missing:
        raise ValueError(
            f"{path}: missing required columns {missing}; available columns are "
            f"{list(table.columns)}."
        )
    if table.empty:
        raise ValueError(f"{path}: table contains no data rows.")
    return table, delimiter
