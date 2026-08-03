#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Align Amazon product metadata with CTGTRec item IDs.

This script replaces the legacy ``2reindex-feat.ipynb`` notebook. It streams an
Amazon metadata JSON-lines file, keeps only ASINs present in
``i_id_mapping.csv``, assigns the corresponding integer ``itemID``, and writes
one metadata row per item in ascending ``itemID`` order.

Both modern JSON-lines metadata and legacy Python-literal metadata are accepted.
The latter is parsed with ``ast.literal_eval`` rather than unsafe ``eval``.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

import numpy as np
import pandas as pd


PREFERRED_COLUMNS = [
    "itemID",
    "asin",
    "title",
    "brand",
    "categories",
    "description",
    "feature",
    "price",
    "imageURL",
    "imageURLHighRes",
    "imUrl",
    "related",
    "also_buy",
    "also_view",
    "salesRank",
]
NESTED_COLUMNS = {
    "categories",
    "description",
    "feature",
    "related",
    "also_buy",
    "also_view",
    "salesRank",
}


def open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def parse_record(line: str, *, path: Path, line_number: int, mode: str) -> dict:
    """Parse one modern JSON object or one legacy dictionary literal."""
    if mode in {"auto", "json"}:
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("metadata record is not an object")
            return value
        except (json.JSONDecodeError, ValueError) as json_error:
            if mode == "json":
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON metadata record."
                ) from json_error

    if mode in {"auto", "python-literal"}:
        try:
            value = ast.literal_eval(line)
        except (SyntaxError, ValueError) as literal_error:
            raise ValueError(
                f"{path}:{line_number}: cannot parse metadata record as JSON "
                "or a Python literal."
            ) from literal_error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: metadata record is not a dict.")
        return value

    raise ValueError(f"Unsupported parser mode: {mode}")


def iter_metadata(path: Path, mode: str) -> Iterator[tuple[int, dict]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, parse_record(
                line,
                path=path,
                line_number=line_number,
                mode=mode,
            )


def load_item_mapping(path: Path) -> pd.DataFrame:
    """Read and validate a tab-separated ASIN-to-itemID mapping."""
    mapping = pd.read_csv(path, sep="\t", dtype={"asin": str})
    required = ["asin", "itemID"]
    missing = [column for column in required if column not in mapping.columns]
    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}; available columns are "
            f"{list(mapping.columns)}."
        )
    mapping = mapping[required].copy()
    mapping["asin"] = mapping["asin"].astype(str).str.strip()
    mapping["itemID"] = pd.to_numeric(mapping["itemID"], errors="raise").astype(
        np.int64
    )
    if mapping["asin"].eq("").any():
        raise ValueError(f"{path}: blank ASIN values are not allowed.")
    if mapping["asin"].duplicated().any():
        duplicates = mapping.loc[mapping["asin"].duplicated(), "asin"].head(20)
        raise ValueError(f"{path}: duplicate ASINs: {duplicates.tolist()}")
    if mapping["itemID"].duplicated().any():
        duplicates = mapping.loc[mapping["itemID"].duplicated(), "itemID"].head(20)
        raise ValueError(f"{path}: duplicate itemIDs: {duplicates.tolist()}")

    mapping = mapping.sort_values("itemID", kind="mergesort").reset_index(drop=True)
    expected = np.arange(len(mapping), dtype=np.int64)
    if not np.array_equal(mapping["itemID"].to_numpy(), expected):
        raise ValueError(f"{path}: itemID must be contiguous from zero.")
    return mapping


def json_cell(value: Any) -> Any:
    """Serialize nested metadata deterministically for CSV storage."""
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def normalize_record(record: dict, *, asin: str, item_id: int) -> dict[str, Any]:
    """Normalize old/new Amazon metadata names without discarding provenance."""
    image_url = record.get("imageURL", record.get("imUrl", ""))
    high_res = record.get("imageURLHighRes", "")
    row: dict[str, Any] = {
        "itemID": item_id,
        "asin": asin,
        "title": record.get("title", ""),
        "brand": record.get("brand", ""),
        "categories": record.get("categories", []),
        "description": record.get("description", ""),
        "feature": record.get("feature", []),
        "price": record.get("price", ""),
        "imageURL": image_url,
        "imageURLHighRes": high_res,
        "imUrl": record.get("imUrl", image_url),
        "related": record.get("related", {}),
        "also_buy": record.get("also_buy", []),
        "also_view": record.get("also_view", []),
        "salesRank": record.get("salesRank", {}),
    }
    for column in NESTED_COLUMNS:
        row[column] = json_cell(row[column])
    return row


def align_metadata(
    metadata_path: Path,
    mapping: pd.DataFrame,
    *,
    parser_mode: str,
    missing_policy: str,
) -> tuple[pd.DataFrame, list[int]]:
    asin_to_item = dict(zip(mapping["asin"], mapping["itemID"], strict=True))
    matched: dict[int, dict[str, Any]] = {}
    duplicate_metadata = 0
    parsed_records = 0

    for _, record in iter_metadata(metadata_path, parser_mode):
        parsed_records += 1
        raw_asin = record.get("asin")
        if raw_asin is None:
            continue
        asin = str(raw_asin).strip()
        item_id = asin_to_item.get(asin)
        if item_id is None:
            continue
        integer_id = int(item_id)
        if integer_id in matched:
            duplicate_metadata += 1
            continue
        matched[integer_id] = normalize_record(
            record,
            asin=asin,
            item_id=integer_id,
        )
        if len(matched) == len(mapping):
            break

    missing_ids = sorted(set(range(len(mapping))) - set(matched))
    if missing_ids and missing_policy == "error":
        raise ValueError(
            f"Missing metadata for {len(missing_ids)} mapped items. "
            f"Example itemIDs: {missing_ids[:20]}. Use --missing_policy empty "
            "only when empty metadata rows are acceptable."
        )

    if missing_ids:
        asin_by_id = dict(zip(mapping["itemID"], mapping["asin"], strict=True))
        for item_id in missing_ids:
            matched[item_id] = normalize_record(
                {},
                asin=str(asin_by_id[item_id]),
                item_id=item_id,
            )

    rows = [matched[item_id] for item_id in range(len(mapping))]
    aligned = pd.DataFrame(rows, columns=PREFERRED_COLUMNS)
    print(f"Parsed metadata records          : {parsed_records}")
    print(f"Matched mapped items             : {len(mapping) - len(missing_ids)}")
    print(f"Duplicate matched metadata rows  : {duplicate_metadata}")
    print(f"Missing mapped items             : {len(missing_ids)}")
    return aligned, missing_ids


def atomic_write_csv(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(output)


def process(args: argparse.Namespace) -> None:
    mapping_path = args.item_mapping.resolve()
    metadata_path = args.metadata.resolve()
    output = args.output.resolve()
    missing_output = args.missing_output.resolve() if args.missing_output else None

    if not mapping_path.is_file():
        raise FileNotFoundError(f"Item mapping not found: {mapping_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    targets = [output] + ([missing_output] if missing_output else [])
    existing = [path for path in targets if path is not None and path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    mapping = load_item_mapping(mapping_path)
    aligned, missing_ids = align_metadata(
        metadata_path,
        mapping,
        parser_mode=args.parser,
        missing_policy=args.missing_policy,
    )
    atomic_write_csv(aligned, output)

    if missing_output is not None:
        missing_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = missing_output.with_name(missing_output.name + ".tmp")
        pd.DataFrame({"itemID": missing_ids}).to_csv(temporary, index=False)
        temporary.replace(missing_output)

    print("\n" + "=" * 72)
    print(f"Item mapping     : {mapping_path}")
    print(f"Raw metadata     : {metadata_path}")
    print(f"Aligned rows     : {len(aligned)}")
    print(f"Output metadata  : {output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item_mapping", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parser",
        choices=["auto", "json", "python-literal"],
        default="auto",
    )
    parser.add_argument(
        "--missing_policy",
        choices=["error", "empty"],
        default="error",
    )
    parser.add_argument(
        "--missing_output",
        type=Path,
        default=None,
        help="Optional CSV containing itemIDs with no matching metadata.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()
