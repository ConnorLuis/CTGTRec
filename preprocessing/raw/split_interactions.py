#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply CTGTRec's strict temporal split to one interaction file.

This file replaces the legacy ``1splitting.ipynb`` notebook. The notebook used
random shuffling and ratio-based labels, which are not part of the CTGTRec
protocol. This command delegates to the canonical implementation in:

    preprocessing/build_temporal_split_inter.py

Run it from the repository root, for example:

    python preprocessing/raw/split_interactions.py \
        --input data/baby/baby.inter \
        --output data/baby/baby_temporal.inter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PREPROCESSING_DIR = Path(__file__).resolve().parents[1]
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

try:
    from build_temporal_split_inter import (  # type: ignore[import-not-found]
        build_temporal_split,
        print_summary,
        read_inter,
    )
except ImportError as exc:  # pragma: no cover - exercised only on bad placement
    raise ImportError(
        "Cannot import preprocessing/build_temporal_split_inter.py. Place this "
        "file at preprocessing/raw/split_interactions.py inside the CTGTRec "
        "repository."
    ) from exc


def process(input_path: Path, output_path: Path, overwrite: bool) -> None:
    source = input_path.resolve()
    output = output_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input interaction file not found: {source}")
    if source == output:
        raise ValueError("Input and output paths must be different.")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Use --overwrite to regenerate."
        )

    original = read_inter(source)
    temporal = build_temporal_split(original)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporal.to_csv(temporary, sep="\t", index=False)
    temporary.replace(output)

    dataset = output.stem.removesuffix("_temporal")
    print_summary(dataset, source, output, temporal)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    process(args.input, args.output, args.overwrite)


if __name__ == "__main__":
    main()
