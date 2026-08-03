#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the strict per-user temporal split used by CTGTRec.

The script reads:

    data/<dataset>/<dataset>.inter

and writes:

    data/<dataset>/<dataset>_temporal.inter

Only ``x_label`` values are regenerated. User IDs, item IDs, ratings, and
timestamps are not re-indexed or otherwise modified. Output rows are written in
per-user chronological order.

Split protocol
--------------
For each user, interactions are ordered by:

    (timestamp, original_record_order)

The original record order is used only to break exact timestamp ties.

Labels are assigned as follows:

    all earlier interactions    -> train (x_label = 0)
    second-to-last interaction  -> valid (x_label = 1)
    last interaction            -> test  (x_label = 2)

CTGTRec uses 5-core datasets, so every user is expected to have at least three
interactions. The script fails instead of silently applying a different split
to shorter histories.

Example
-------
python preprocessing/build_temporal_split_inter.py \
    --data_root data \
    --datasets baby sports clothing microlens
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_COLS = ["userID", "itemID", "rating", "timestamp", "x_label"]
ORIGINAL_ORDER_COL = "__original_record_order__"


def _coerce_integer_column(df: pd.DataFrame, column: str, path: Path) -> None:
    """Validate and convert an identifier column to int64 without truncation."""
    numeric = pd.to_numeric(df[column], errors="raise")

    if numeric.isna().any():
        raise ValueError(f"{path}: column {column!r} contains missing values.")

    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: column {column!r} contains non-finite values.")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{path}: column {column!r} must contain integer IDs.")
    if (values < 0).any():
        raise ValueError(f"{path}: column {column!r} contains negative IDs.")

    df[column] = numeric.astype(np.int64)


def read_inter(path: Path) -> pd.DataFrame:
    """Read and validate an MMRec-format interaction file."""
    if not path.is_file():
        raise FileNotFoundError(f"Input interaction file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    missing = [column for column in REQUIRED_COLS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required columns {missing}; "
            f"available columns are {list(df.columns)}."
        )

    df = df[REQUIRED_COLS].copy()

    if df.empty:
        raise ValueError(f"{path}: interaction file is empty.")
    if df[REQUIRED_COLS].isna().any().any():
        missing_counts = df[REQUIRED_COLS].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"{path}: missing values found: {missing_counts}")

    _coerce_integer_column(df, "userID", path)
    _coerce_integer_column(df, "itemID", path)

    timestamp = pd.to_numeric(df["timestamp"], errors="raise")
    timestamp_values = timestamp.to_numpy(dtype=np.float64)
    if not np.isfinite(timestamp_values).all():
        raise ValueError(f"{path}: timestamp contains non-finite values.")
    df["timestamp"] = timestamp

    return df


def build_temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the paper's strict per-user chronological leave-one-out split.

    Exact timestamp ties are resolved by the row order in the input file.
    """
    work = df.copy()
    work[ORIGINAL_ORDER_COL] = np.arange(len(work), dtype=np.int64)

    # Do not include itemID as a tie-breaking key. The paper uses the original
    # record order when timestamps are equal.
    work = work.sort_values(
        ["userID", "timestamp", ORIGINAL_ORDER_COL],
        kind="mergesort",
    ).reset_index(drop=True)

    group_sizes = work.groupby("userID", sort=False)["userID"].transform("size")
    short_mask = group_sizes < 3
    if short_mask.any():
        short_users = (
            work.loc[short_mask, ["userID"]]
            .drop_duplicates()
            .head(20)["userID"]
            .astype(int)
            .tolist()
        )
        total_short_users = int(work.loc[short_mask, "userID"].nunique())
        raise ValueError(
            "Strict CTGTRec splitting requires at least three interactions per "
            f"user, but found {total_short_users} shorter user histories. "
            f"Example user IDs: {short_users}"
        )

    position = work.groupby("userID", sort=False).cumcount()

    work["x_label"] = np.int8(0)
    work.loc[position == group_sizes - 2, "x_label"] = np.int8(1)
    work.loc[position == group_sizes - 1, "x_label"] = np.int8(2)

    validate_temporal_split(work)
    return work.drop(columns=[ORIGINAL_ORDER_COL])[REQUIRED_COLS].reset_index(drop=True)


def validate_temporal_split(work: pd.DataFrame) -> None:
    """Validate label counts, ordering, and temporal boundaries per user."""
    label_values = set(work["x_label"].astype(int).unique().tolist())
    if not label_values.issubset({0, 1, 2}):
        raise RuntimeError(f"Unexpected x_label values: {sorted(label_values)}")

    grouped = work.groupby("userID", sort=False)
    group_sizes = grouped.size()

    label_counts = (
        work.groupby(["userID", "x_label"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )

    expected_train = group_sizes - 2
    invalid_counts = (
        (label_counts[0] != expected_train)
        | (label_counts[1] != 1)
        | (label_counts[2] != 1)
    )
    if invalid_counts.any():
        bad_users = invalid_counts[invalid_counts].index[:20].tolist()
        raise RuntimeError(
            "Invalid train/valid/test counts after splitting. "
            f"Example user IDs: {bad_users}"
        )

    position = grouped.cumcount()
    repeated_sizes = grouped["userID"].transform("size")
    expected_labels = np.zeros(len(work), dtype=np.int8)
    expected_labels[position.to_numpy() == (repeated_sizes.to_numpy() - 2)] = 1
    expected_labels[position.to_numpy() == (repeated_sizes.to_numpy() - 1)] = 2

    if not np.array_equal(work["x_label"].to_numpy(dtype=np.int8), expected_labels):
        raise RuntimeError("x_label order is inconsistent with strict leave-one-out.")

    train_max = work[work["x_label"] == 0].groupby("userID")["timestamp"].max()
    valid_time = work[work["x_label"] == 1].set_index("userID")["timestamp"]
    test_time = work[work["x_label"] == 2].set_index("userID")["timestamp"]

    temporal_violation = (train_max > valid_time) | (valid_time > test_time)
    if temporal_violation.any():
        bad_users = temporal_violation[temporal_violation].index[:20].tolist()
        raise RuntimeError(
            "Per-user temporal order validation failed. "
            f"Example user IDs: {bad_users}"
        )


def print_summary(dataset: str, source: Path, output: Path, df: pd.DataFrame) -> None:
    """Print a concise reproducibility summary."""
    label_counts = (
        df["x_label"]
        .value_counts()
        .reindex([0, 1, 2], fill_value=0)
        .astype(int)
    )
    tied_rows = int(
        df.duplicated(subset=["userID", "timestamp"], keep=False).sum()
    )

    print("\n" + "=" * 72)
    print(f"Dataset          : {dataset}")
    print(f"Input            : {source}")
    print(f"Output           : {output}")
    print(f"Users            : {df['userID'].nunique()}")
    print(f"Items            : {df['itemID'].nunique()}")
    print(f"Interactions     : {len(df)}")
    print(f"Train / Valid / Test: {label_counts[0]} / {label_counts[1]} / {label_counts[2]}")
    print(f"Rows in timestamp ties: {tied_rows}")
    print("Temporal check   : passed")


def process_dataset(
    data_root: Path,
    dataset: str,
    input_suffix: str,
    output_suffix: str,
    overwrite: bool,
) -> Path:
    """Process one dataset and return the generated file path."""
    dataset_dir = data_root / dataset
    source = dataset_dir / f"{dataset}{input_suffix}"
    output = dataset_dir / f"{dataset}{output_suffix}"

    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must be different.")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. "
            "Use --overwrite to regenerate it."
        )

    original = read_inter(source)
    temporal = build_temporal_split(original)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporal.to_csv(temporary, sep="\t", index=False)
    temporary.replace(output)

    print_summary(dataset, source, output, temporal)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate MMRec x_label values using CTGTRec's strict per-user "
            "chronological leave-one-out protocol."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="Root directory containing one subdirectory per dataset.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset directory names, e.g. baby sports clothing microlens.",
    )
    parser.add_argument(
        "--input_suffix",
        default=".inter",
        help="Input suffix appended to each dataset name (default: .inter).",
    )
    parser.add_argument(
        "--output_suffix",
        default="_temporal.inter",
        help=(
            "Output suffix appended to each dataset name "
            "(default: _temporal.inter)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing derived temporal interaction file.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        process_dataset(
            data_root=args.data_root,
            dataset=dataset,
            input_suffix=args.input_suffix,
            output_suffix=args.output_suffix,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
