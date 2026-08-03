#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Encode Amazon text features and align precomputed image features.

This script replaces the legacy ``3feat-encoder.ipynb`` notebook while
preserving its released feature protocol:

* text = title + brand + first category path + description;
* text encoder = ``sentence-transformers/all-MiniLM-L6-v2`` (384 dimensions);
* image input = Amazon binary records containing a 10-byte ASIN followed by
  4096 float32 values;
* missing image features are filled with the mean of available mapped features.

Use ``--mode text``, ``--mode image`` or ``--mode all``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd


WHITESPACE = re.compile(r"\s+")


def parse_nested(value: Any) -> Any:
    """Parse JSON/Python-literal cells produced by old or new metadata code."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, dict, tuple)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text


def stringify(value: Any) -> str:
    """Convert scalar/list metadata to readable text."""
    parsed = parse_nested(value)
    if parsed is None:
        return ""
    if isinstance(parsed, dict):
        return " ".join(stringify(item) for item in parsed.values())
    if isinstance(parsed, (list, tuple)):
        return " ".join(stringify(item) for item in parsed)
    return str(parsed)


def first_category_path(value: Any) -> str:
    """Match the notebook: concatenate only the first category path."""
    parsed = parse_nested(value)
    if not isinstance(parsed, list) or not parsed:
        return stringify(parsed)
    first = parsed[0]
    if isinstance(first, (list, tuple)):
        return " ".join(str(part) for part in first)
    return stringify(first)


def clean_text(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip()


def load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, dtype={"asin": str})
    required = ["itemID", "asin", "title", "brand", "categories", "description"]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}; available columns are "
            f"{list(metadata.columns)}."
        )

    metadata["itemID"] = pd.to_numeric(metadata["itemID"], errors="raise").astype(
        np.int64
    )
    metadata["asin"] = metadata["asin"].astype(str).str.strip()
    metadata = metadata.sort_values("itemID", kind="mergesort").reset_index(drop=True)
    expected = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata["itemID"].to_numpy(), expected):
        raise ValueError(f"{path}: itemID must be unique and contiguous from zero.")
    if metadata["asin"].eq("").any() or metadata["asin"].duplicated().any():
        raise ValueError(f"{path}: ASIN values must be non-empty and unique.")
    return metadata


def build_sentences(metadata: pd.DataFrame, *, include_feature: bool) -> list[str]:
    sentences: list[str] = []
    for row in metadata.itertuples(index=False):
        parts = [
            stringify(getattr(row, "title", "")),
            stringify(getattr(row, "brand", "")),
            first_category_path(getattr(row, "categories", "")),
        ]
        if include_feature and hasattr(row, "feature"):
            parts.append(stringify(getattr(row, "feature")))
        parts.append(stringify(getattr(row, "description", "")))
        sentences.append(clean_text(" ".join(part for part in parts if part)))
    return sentences


def encode_text_features(
    metadata: pd.DataFrame,
    *,
    output: Path,
    model_name: str,
    batch_size: int,
    device: str | None,
    expected_dimension: int | None,
    include_feature: bool,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Text encoding requires sentence-transformers. Install it with "
            "`pip install sentence-transformers`."
        ) from exc

    sentences = build_sentences(metadata, include_feature=include_feature)
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)

    if embeddings.shape[0] != len(metadata):
        raise RuntimeError("Text feature row count does not match item count.")
    if expected_dimension is not None and embeddings.shape[1] != expected_dimension:
        raise RuntimeError(
            f"Expected text dimension {expected_dimension}, "
            f"received {embeddings.shape[1]} from {model_name}."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npy")
    np.save(temporary, embeddings)
    temporary.replace(output)
    print(f"Text features : {embeddings.shape} -> {output}")
    return embeddings


def iter_image_binary(
    path: Path,
    *,
    asin_bytes: int,
    feature_dimension: int,
) -> Iterator[tuple[str, np.ndarray]]:
    """Stream Amazon image records without loading the full source file."""
    feature_bytes = feature_dimension * np.dtype("<f4").itemsize
    with path.open("rb") as handle:
        record_number = 0
        while True:
            asin_raw = handle.read(asin_bytes)
            if not asin_raw:
                return
            record_number += 1
            if len(asin_raw) != asin_bytes:
                raise ValueError(
                    f"{path}: truncated ASIN at image record {record_number}."
                )
            payload = handle.read(feature_bytes)
            if len(payload) != feature_bytes:
                raise ValueError(
                    f"{path}: truncated feature vector at image record "
                    f"{record_number}."
                )
            try:
                asin = asin_raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid ASIN bytes at image record {record_number}."
                ) from exc
            vector = np.frombuffer(payload, dtype="<f4").copy()
            yield asin, vector


def encode_image_features(
    metadata: pd.DataFrame,
    *,
    binary_path: Path,
    output: Path,
    missing_output: Path,
    asin_bytes: int,
    feature_dimension: int,
) -> np.ndarray:
    if not binary_path.is_file():
        raise FileNotFoundError(f"Image feature binary not found: {binary_path}")

    item_by_asin = dict(zip(metadata["asin"], metadata["itemID"], strict=True))
    item_count = len(metadata)
    features = np.empty((item_count, feature_dimension), dtype=np.float32)
    found = np.zeros(item_count, dtype=bool)
    vector_sum = np.zeros(feature_dimension, dtype=np.float64)
    matched_count = 0

    for asin, vector in iter_image_binary(
        binary_path,
        asin_bytes=asin_bytes,
        feature_dimension=feature_dimension,
    ):
        item_id = item_by_asin.get(asin)
        if item_id is None or found[int(item_id)]:
            continue
        integer_id = int(item_id)
        features[integer_id] = vector
        found[integer_id] = True
        vector_sum += vector
        matched_count += 1

    if matched_count == 0:
        raise ValueError(
            "None of the mapped ASINs were found in the image feature binary. "
            "Check the dataset category, ASIN mapping and binary format."
        )

    mean_vector = (vector_sum / matched_count).astype(np.float32)
    missing_ids = np.flatnonzero(~found)
    features[missing_ids] = mean_vector

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npy")
    np.save(temporary, features)
    temporary.replace(output)

    missing_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_missing = missing_output.with_name(missing_output.name + ".tmp")
    np.savetxt(temporary_missing, missing_ids, delimiter=",", fmt="%d")
    temporary_missing.replace(missing_output)

    print(f"Matched image items : {matched_count}/{item_count}")
    print(f"Mean-filled items   : {len(missing_ids)}")
    print(f"Image features      : {features.shape} -> {output}")
    print(f"Missing item IDs    : {missing_output}")
    return features


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def process(args: argparse.Namespace) -> None:
    metadata_path = args.metadata.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Aligned metadata not found: {metadata_path}")
    metadata = load_metadata(metadata_path)
    output_dir = args.output_dir.resolve()

    text_output = output_dir / args.text_output_name
    image_output = output_dir / args.image_output_name
    missing_output = output_dir / args.missing_output_name

    targets: list[Path] = []
    if args.mode in {"text", "all"}:
        targets.append(text_output)
    if args.mode in {"image", "all"}:
        targets.extend([image_output, missing_output])
    ensure_writable(targets, args.overwrite)

    if args.mode in {"text", "all"}:
        encode_text_features(
            metadata,
            output=text_output,
            model_name=args.text_model,
            batch_size=args.batch_size,
            device=args.device,
            expected_dimension=args.expected_text_dimension,
            include_feature=args.include_feature,
        )

    if args.mode in {"image", "all"}:
        if args.image_binary is None:
            raise ValueError("--image_binary is required for image/all mode.")
        encode_image_features(
            metadata,
            binary_path=args.image_binary.resolve(),
            output=image_output,
            missing_output=missing_output,
            asin_bytes=args.asin_bytes,
            feature_dimension=args.image_dimension,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["text", "image", "all"], required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--text_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--device",
        default=None,
        help="SentenceTransformer device, e.g. cuda, cuda:0 or cpu.",
    )
    parser.add_argument("--expected_text_dimension", type=int, default=384)
    parser.add_argument(
        "--include_feature",
        action="store_true",
        help=(
            "Append the newer Amazon bullet-point `feature` field. Disabled by "
            "default to reproduce the legacy notebook text composition."
        ),
    )
    parser.add_argument("--image_binary", type=Path, default=None)
    parser.add_argument("--asin_bytes", type=int, default=10)
    parser.add_argument("--image_dimension", type=int, default=4096)
    parser.add_argument("--text_output_name", default="text_feat.npy")
    parser.add_argument("--image_output_name", default="image_feat.npy")
    parser.add_argument(
        "--missing_output_name",
        default="missed_img_itemIDs.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    process(parse_args())


if __name__ == "__main__":
    main()
