#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build re-indexed MMRec interaction files from original raw datasets.

This script replaces the legacy ``0rating2inter.ipynb`` notebook. It supports:

* Amazon ratings-only CSV files;
* Amazon review JSON-lines files (optionally gzip-compressed);
* MicroLens interaction CSV/TSV files.

The script removes rows with missing identifiers/timestamps, removes duplicate
``(user, item, timestamp)`` interactions, applies iterative user/item k-core
filtering, re-indexes users and items to contiguous zero-based IDs, and writes:

* ``<dataset>.inter``;
* ``u_id_mapping.csv``;
* ``i_id_mapping.csv``;
* ``raw_preprocessing_manifest.json``.

The generated ``x_label`` column is a placeholder containing zeros. Run the
strict temporal splitter afterwards.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterator, Sequence, TextIO

import numpy as np
import pandas as pd


INTER_COLUMNS = ["userID", "itemID", "rating", "timestamp", "x_label"]
ORIGINAL_ORDER = "__original_record_order__"


def open_text(path: Path) -> TextIO:
    """Open plain-text or gzip-compressed input as UTF-8 text."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_amazon_ratings_csv(
    path: Path,
    *,
    delimiter: str,
    column_order: str,
) -> pd.DataFrame:
    """Read a headerless Amazon ratings-only CSV without losing ASIN zeros."""
    raw = pd.read_csv(
        path,
        sep=delimiter,
        header=None,
        dtype=str,
        keep_default_na=False,
        compression="infer",
    )
    if raw.shape[1] != 4:
        raise ValueError(
            f"{path}: expected four columns, found {raw.shape[1]}. "
            "Check --delimiter and --amazon_column_order."
        )

    if column_order == "user-item-rating-timestamp":
        raw.columns = ["raw_user_id", "raw_item_id", "rating", "timestamp"]
    elif column_order == "item-user-rating-timestamp":
        raw.columns = ["raw_item_id", "raw_user_id", "rating", "timestamp"]
    else:
        raise ValueError(f"Unsupported Amazon column order: {column_order}")

    return raw


def iter_json_lines(path: Path) -> Iterator[tuple[int, dict]]:
    """Yield non-empty JSON objects with one-based source line numbers."""
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object, "
                    f"found {type(record).__name__}."
                )
            yield line_number, record


def read_amazon_reviews_jsonl(path: Path) -> pd.DataFrame:
    """Read Amazon review JSON-lines data using official field names."""
    rows: list[dict[str, object]] = []
    required = ("reviewerID", "asin", "overall", "unixReviewTime")

    for line_number, record in iter_json_lines(path):
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields {missing}.")
        rows.append(
            {
                "raw_user_id": record["reviewerID"],
                "raw_item_id": record["asin"],
                "rating": record["overall"],
                "timestamp": record["unixReviewTime"],
            }
        )

    if not rows:
        raise ValueError(f"{path}: no review records were found.")
    return pd.DataFrame(rows)


def read_microlens_table(
    path: Path,
    *,
    delimiter: str,
    user_column: str,
    item_column: str,
    timestamp_column: str,
    rating_column: str | None,
    implicit_rating: float,
) -> pd.DataFrame:
    """Read a MicroLens interaction CSV/TSV with explicit column names."""
    raw = pd.read_csv(path, sep=delimiter, compression="infer")
    required = [user_column, item_column, timestamp_column]
    if rating_column:
        required.append(rating_column)
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}; available columns are "
            f"{list(raw.columns)}."
        )

    result = raw[[user_column, item_column, timestamp_column]].copy()
    result.columns = ["raw_user_id", "raw_item_id", "timestamp"]
    if rating_column:
        result["rating"] = raw[rating_column]
    else:
        result["rating"] = implicit_rating
    return result[["raw_user_id", "raw_item_id", "rating", "timestamp"]]


def clean_interactions(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Validate numeric fields and preserve source order for later tie-breaking."""
    columns = ["raw_user_id", "raw_item_id", "rating", "timestamp"]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing normalized columns {missing}.")

    work = df[columns].copy()
    work[ORIGINAL_ORDER] = np.arange(len(work), dtype=np.int64)

    user_text = work["raw_user_id"].astype("string").str.strip()
    item_text = work["raw_item_id"].astype("string").str.strip()
    invalid_id = (
        work["raw_user_id"].isna()
        | work["raw_item_id"].isna()
        | user_text.eq("")
        | item_text.eq("")
    )
    missing_timestamp = work["timestamp"].isna()
    removed_missing = int((invalid_id | missing_timestamp).sum())
    work = work.loc[~(invalid_id | missing_timestamp)].copy()
    work["raw_user_id"] = work["raw_user_id"].astype(str).str.strip()
    work["raw_item_id"] = work["raw_item_id"].astype(str).str.strip()

    work["rating"] = pd.to_numeric(work["rating"], errors="raise")
    work["timestamp"] = pd.to_numeric(work["timestamp"], errors="raise")
    rating_values = work["rating"].to_numpy(dtype=np.float64)
    timestamp_values = work["timestamp"].to_numpy(dtype=np.float64)
    if not np.isfinite(rating_values).all():
        raise ValueError(f"{source}: rating contains non-finite values.")
    if not np.isfinite(timestamp_values).all():
        raise ValueError(f"{source}: timestamp contains non-finite values.")

    before = len(work)
    work = work.drop_duplicates(
        subset=["raw_user_id", "raw_item_id", "timestamp"],
        keep="first",
    )
    removed_duplicates = before - len(work)

    if work.empty:
        raise ValueError(f"{source}: no interactions remain after cleaning.")

    print(f"Removed missing/blank ID or timestamp rows : {removed_missing}")
    print(f"Removed duplicate user-item-time rows     : {removed_duplicates}")
    return work.reset_index(drop=True)


def iterative_k_core(
    df: pd.DataFrame,
    *,
    min_user_interactions: int,
    min_item_interactions: int,
) -> tuple[pd.DataFrame, int]:
    """Iteratively filter users/items until both degree constraints hold."""
    if min_user_interactions < 1 or min_item_interactions < 1:
        raise ValueError("k-core thresholds must be positive integers.")

    work = df.copy()
    rounds = 0
    while True:
        user_counts = work["raw_user_id"].value_counts(sort=False)
        item_counts = work["raw_item_id"].value_counts(sort=False)
        keep = (
            work["raw_user_id"].map(user_counts).ge(min_user_interactions)
            & work["raw_item_id"].map(item_counts).ge(min_item_interactions)
        )
        if bool(keep.all()):
            return work.reset_index(drop=True), rounds

        removed = int((~keep).sum())
        work = work.loc[keep].copy()
        rounds += 1
        print(f"k-core round {rounds}: removed {removed} interactions")
        if work.empty:
            raise ValueError(
                "All interactions were removed by k-core filtering. "
                "Check the input data and thresholds."
            )


def ordered_unique(series: pd.Series, order: str) -> list[str]:
    values = pd.unique(series).tolist()
    if order == "appearance":
        return values
    if order == "sorted":
        return sorted(values)
    raise ValueError(f"Unsupported mapping order: {order}")


def reindex_interactions(
    df: pd.DataFrame,
    *,
    mapping_order: str,
    item_mapping_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign contiguous integer IDs while retaining source row order."""
    users = ordered_unique(df["raw_user_id"], mapping_order)
    items = ordered_unique(df["raw_item_id"], mapping_order)
    user_map = {raw_id: index for index, raw_id in enumerate(users)}
    item_map = {raw_id: index for index, raw_id in enumerate(items)}

    ordered = df.sort_values(ORIGINAL_ORDER, kind="mergesort").copy()
    ordered["userID"] = ordered["raw_user_id"].map(user_map).astype(np.int64)
    ordered["itemID"] = ordered["raw_item_id"].map(item_map).astype(np.int64)
    ordered["x_label"] = np.int8(0)
    interactions = ordered[INTER_COLUMNS].reset_index(drop=True)

    user_mapping = pd.DataFrame(
        {"user_id": users, "userID": np.arange(len(users), dtype=np.int64)}
    )
    item_mapping = pd.DataFrame(
        {
            item_mapping_name: items,
            "itemID": np.arange(len(items), dtype=np.int64),
        }
    )
    return interactions, user_mapping, item_mapping


def validate_outputs(
    interactions: pd.DataFrame,
    user_mapping: pd.DataFrame,
    item_mapping: pd.DataFrame,
    *,
    min_user_interactions: int,
    min_item_interactions: int,
) -> None:
    """Validate contiguous mappings, schema and final k-core constraints."""
    if list(interactions.columns) != INTER_COLUMNS:
        raise RuntimeError(f"Unexpected interaction schema: {interactions.columns}")
    if not np.array_equal(
        np.sort(interactions["userID"].unique()),
        np.arange(len(user_mapping)),
    ):
        raise RuntimeError("User IDs are not contiguous from zero.")
    if not np.array_equal(
        np.sort(interactions["itemID"].unique()),
        np.arange(len(item_mapping)),
    ):
        raise RuntimeError("Item IDs are not contiguous from zero.")
    if int(interactions.groupby("userID").size().min()) < min_user_interactions:
        raise RuntimeError("User k-core validation failed.")
    if int(interactions.groupby("itemID").size().min()) < min_item_interactions:
        raise RuntimeError("Item k-core validation failed.")


def atomic_write_dataframe(
    df: pd.DataFrame,
    output: Path,
    *,
    separator: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    df.to_csv(temporary, sep=separator, index=False)
    temporary.replace(output)


def process(args: argparse.Namespace) -> None:
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    if args.source_type == "amazon-ratings-csv":
        if args.amazon_column_order is None:
            raise ValueError(
                "--amazon_column_order is required for headerless ratings CSV "
                "files. Do not guess the user/item column order."
            )
        raw = read_amazon_ratings_csv(
            source,
            delimiter=args.delimiter,
            column_order=args.amazon_column_order,
        )
        item_mapping_name = "asin"
    elif args.source_type == "amazon-reviews-jsonl":
        raw = read_amazon_reviews_jsonl(source)
        item_mapping_name = "asin"
    elif args.source_type == "microlens-csv":
        raw = read_microlens_table(
            source,
            delimiter=args.delimiter,
            user_column=args.user_column,
            item_column=args.item_column,
            timestamp_column=args.timestamp_column,
            rating_column=args.rating_column,
            implicit_rating=args.implicit_rating,
        )
        item_mapping_name = "videoID"
    else:
        raise ValueError(f"Unsupported source type: {args.source_type}")

    print(f"Raw interactions : {len(raw)}")
    cleaned = clean_interactions(raw, source)
    filtered, rounds = iterative_k_core(
        cleaned,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    interactions, user_mapping, item_mapping = reindex_interactions(
        filtered,
        mapping_order=args.mapping_order,
        item_mapping_name=item_mapping_name,
    )
    validate_outputs(
        interactions,
        user_mapping,
        item_mapping,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )

    output_dir = args.output_dir.resolve()
    outputs = {
        "interaction": output_dir / f"{args.dataset}.inter",
        "user_mapping": output_dir / "u_id_mapping.csv",
        "item_mapping": output_dir / "i_id_mapping.csv",
        "manifest": output_dir / "raw_preprocessing_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite to regenerate:\n"
            + formatted
        )

    atomic_write_dataframe(interactions, outputs["interaction"], separator="\t")
    # Mapping files use the delimiter implied by their .csv extension. Readers
    # remain backward-compatible with legacy tab-separated mapping files.
    atomic_write_dataframe(user_mapping, outputs["user_mapping"], separator=",")
    atomic_write_dataframe(item_mapping, outputs["item_mapping"], separator=",")

    manifest = {
        "source_type": args.source_type,
        "source_file": str(source),
        "dataset": args.dataset,
        "mapping_order": args.mapping_order,
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "k_core_rounds": rounds,
        "num_interactions": int(len(interactions)),
        "num_users": int(len(user_mapping)),
        "num_items": int(len(item_mapping)),
        "interaction_file": outputs["interaction"].name,
        "user_mapping_file": outputs["user_mapping"].name,
        "item_mapping_file": outputs["item_mapping"].name,
        "interaction_separator": "tab",
        "mapping_separator": "comma",
        "x_label_note": (
            "Placeholder zeros only. Run the strict per-user temporal splitter."
        ),
    }
    outputs["manifest"].parent.mkdir(parents=True, exist_ok=True)
    temporary = outputs["manifest"].with_name(outputs["manifest"].name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(outputs["manifest"])

    print("\n" + "=" * 72)
    print(f"Source             : {source}")
    print(f"Source type        : {args.source_type}")
    print(f"k-core rounds      : {rounds}")
    print(f"Users              : {len(user_mapping)}")
    print(f"Items              : {len(item_mapping)}")
    print(f"Interactions       : {len(interactions)}")
    print(f"Interaction output : {outputs['interaction']}")
    print(f"User mapping       : {outputs['user_mapping']}")
    print(f"Item mapping       : {outputs['item_mapping']}")
    print("Next step          : run the strict temporal splitter")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_type",
        choices=[
            "amazon-ratings-csv",
            "amazon-reviews-jsonl",
            "microlens-csv",
        ],
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Output stem, e.g. baby, sports, clothing or microlens.",
    )
    parser.add_argument("--delimiter", default=",")
    parser.add_argument(
        "--amazon_column_order",
        choices=[
            "user-item-rating-timestamp",
            "item-user-rating-timestamp",
        ],
        default=None,
        help=(
            "Required for amazon-ratings-csv. The legacy 2014 notebook used "
            "user-item-rating-timestamp; the newer Amazon page documents "
            "item-user-rating-timestamp."
        ),
    )
    parser.add_argument("--user_column", default="userID")
    parser.add_argument("--item_column", default="videoID")
    parser.add_argument("--timestamp_column", default="timestamp")
    parser.add_argument("--rating_column", default=None)
    parser.add_argument("--implicit_rating", type=float, default=1.0)
    parser.add_argument("--min_user_interactions", type=int, default=5)
    parser.add_argument("--min_item_interactions", type=int, default=5)
    parser.add_argument(
        "--mapping_order",
        choices=["appearance", "sorted"],
        default="appearance",
        help="Use first appearance for legacy-compatible ID assignment.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()
