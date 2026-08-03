#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_temporal_split_inter.py

Purpose
-------
Re-generate only the x_label column of MMRec-format .inter files using a
per-user chronological split, while keeping userID / itemID / rating /
timestamp unchanged. This preserves existing u_id_mapping.csv,
i_id_mapping.csv, image_feat.npy, text_feat.npy, etc.

Input format
------------
userID\titemID\trating\ttimestamp\tx_label

Output format
-------------
userID\titemID\trating\ttimestamp\tx_label

x_label convention
------------------
0 = train, 1 = valid, 2 = test

Recommended first use
---------------------
# Create new dataset folders data/baby, data/sports, ...
python preprocessing/build_temporal_split_inter.py \
  --data_root data \
  --datasets baby sports clothing microlens \
  --split_mode leave_one_out \
  --output_style new_dataset \
  --copy_side_files

Then rebuild equal-time snapshots and adjacency files on the *_temporal datasets.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

REQUIRED_COLS = ["userID", "itemID", "rating", "timestamp", "x_label"]


def read_inter(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input .inter file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}. Columns={list(df.columns)}")

    # Keep only the standard columns and preserve their order.
    df = df[REQUIRED_COLS].copy()

    # Enforce stable numeric dtypes where possible.
    df["userID"] = df["userID"].astype(int)
    df["itemID"] = df["itemID"].astype(int)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise")
    df["x_label"] = df["x_label"].astype(int)
    return df


def assign_leave_one_out(g: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user chronological leave-one-out split.

    n >= 3: last -> test, second last -> valid, earlier -> train
    n == 2: last -> test, first -> train
    n == 1: only train

    For 5-core datasets, n is normally >= 5, so every user gets train/valid/test.
    """
    n = len(g)
    labels = [0] * n
    if n >= 3:
        labels[-2] = 1
        labels[-1] = 2
    elif n == 2:
        labels[-1] = 2
    g = g.copy()
    g["x_label"] = labels
    return g


def assign_ratio(g: pd.DataFrame, train_ratio: float, valid_ratio: float) -> pd.DataFrame:
    """
    Per-user chronological ratio split.

    For n >= 3, ensure at least one valid and one test interaction.
    For n < 3, fall back to the leave-one-out behavior.
    """
    n = len(g)
    if n < 3:
        return assign_leave_one_out(g)

    train_len = int(n * train_ratio)
    valid_len = int(n * valid_ratio)

    # Ensure non-empty valid/test and at least one training interaction.
    train_len = max(1, min(train_len, n - 2))
    valid_len = max(1, min(valid_len, n - train_len - 1))
    test_len = n - train_len - valid_len
    if test_len < 1:
        test_len = 1
        if valid_len > 1:
            valid_len -= 1
        else:
            train_len -= 1

    labels = [0] * train_len + [1] * valid_len + [2] * test_len
    if len(labels) != n:
        raise RuntimeError(f"Internal split error: len(labels)={len(labels)} != n={n}")

    g = g.copy()
    g["x_label"] = labels
    return g


def build_temporal_split(
    df: pd.DataFrame,
    split_mode: str,
    train_ratio: float,
    valid_ratio: float,
) -> pd.DataFrame:
    # Stable chronological order inside each user. __orig_order__ is used only to
    # break exact timestamp ties deterministically.
    df = df.copy()
    df["__orig_order__"] = range(len(df))
    df = df.sort_values(["userID", "timestamp", "itemID", "__orig_order__"]).reset_index(drop=True)

    parts: List[pd.DataFrame] = []
    for _, g in df.groupby("userID", sort=False):
        if split_mode == "leave_one_out":
            parts.append(assign_leave_one_out(g))
        elif split_mode == "ratio":
            parts.append(assign_ratio(g, train_ratio=train_ratio, valid_ratio=valid_ratio))
        else:
            raise ValueError(f"Unsupported split_mode={split_mode}")

    out = pd.concat(parts, axis=0).sort_values(["userID", "timestamp", "itemID", "__orig_order__"])
    out = out.drop(columns=["__orig_order__"]).reset_index(drop=True)
    out["x_label"] = out["x_label"].astype(int)
    return out[REQUIRED_COLS]


def temporal_check(df: pd.DataFrame) -> Tuple[bool, int, int, float]:
    train = df[df["x_label"] == 0]
    valid = df[df["x_label"] == 1]
    test = df[df["x_label"] == 2]

    global_ok = False
    if len(train) > 0 and len(valid) > 0 and len(test) > 0:
        global_ok = bool(
            train["timestamp"].max() <= valid["timestamp"].min()
            and valid["timestamp"].max() <= test["timestamp"].min()
        )

    bad_users = 0
    checked_users = 0
    for _, g in df.groupby("userID"):
        tr = g[g["x_label"] == 0]["timestamp"]
        va = g[g["x_label"] == 1]["timestamp"]
        te = g[g["x_label"] == 2]["timestamp"]
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            continue
        checked_users += 1
        if not (tr.max() <= va.min() and va.max() <= te.min()):
            bad_users += 1

    violation_ratio = bad_users / checked_users if checked_users else 0.0
    return global_ok, checked_users, bad_users, violation_ratio


def print_stats(name: str, df: pd.DataFrame) -> None:
    print(f"\n[Dataset] {name}")
    print(f"Total interactions: {len(df)}")
    print("Label counts:")
    print(df["x_label"].value_counts().sort_index().to_string())

    for label, label_name in [(0, "train"), (1, "valid"), (2, "test")]:
        part = df[df["x_label"] == label]
        if len(part) == 0:
            print(f"{label_name}: EMPTY")
        else:
            print(
                f"{label_name}: count={len(part)}, "
                f"ts_min={part['timestamp'].min()}, ts_max={part['timestamp'].max()}"
            )

    global_ok, checked_users, bad_users, ratio = temporal_check(df)
    print(f"Global temporal split OK: {global_ok}")
    print(
        "Per-user temporal check: "
        f"checked_users={checked_users}, bad_users={bad_users}, violation_ratio={ratio:.6f}"
    )
    print(
        "Note: per-user chronological split usually does NOT require global_ok=True, "
        "because different users have different time ranges. The key target is "
        "per-user violation_ratio=0."
    )


def copy_side_files(src_dir: Path, dst_dir: Path, src_inter_name: str) -> None:
    """Copy side files so *_temporal can be used as an MMRec dataset directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    skip_suffixes = {".inter"}
    skip_dir_keywords = {
        "dynamic_snapshots",
        "time_interval_snapshots",
        "__pycache__",
    }

    for p in src_dir.iterdir():
        if p.is_dir():
            # Do not copy old snapshot directories; they must be rebuilt.
            if any(k in p.name for k in skip_dir_keywords):
                continue
            continue

        if p.name == src_inter_name:
            continue
        if p.suffix in skip_suffixes:
            continue

        target = dst_dir / p.name
        if not target.exists():
            shutil.copy2(p, target)


def resolve_paths(data_root: Path, dataset: str, output_style: str) -> Tuple[Path, Path, Path, str]:
    src_dir = data_root / dataset
    src_inter = src_dir / f"{dataset}.inter"

    if output_style == "same_dir":
        dst_dir = src_dir
        out_dataset_name = dataset
        dst_inter = dst_dir / f"{dataset}_temporal.inter"
    elif output_style == "new_dataset":
        out_dataset_name = f"{dataset}_temporal"
        dst_dir = data_root / out_dataset_name
        dst_inter = dst_dir / f"{out_dataset_name}.inter"
    elif output_style == "overwrite":
        dst_dir = src_dir
        out_dataset_name = dataset
        dst_inter = src_inter
    else:
        raise ValueError(f"Unsupported output_style={output_style}")

    return src_inter, dst_dir, dst_inter, out_dataset_name


def process_dataset(args: argparse.Namespace, dataset: str) -> None:
    data_root = Path(args.data_root)
    src_inter, dst_dir, dst_inter, out_dataset_name = resolve_paths(
        data_root, dataset, args.output_style
    )

    print("\n" + "=" * 80)
    print(f"Input dataset : {dataset}")
    print(f"Input inter   : {src_inter}")
    print(f"Output dataset: {out_dataset_name}")
    print(f"Output inter  : {dst_inter}")

    df = read_inter(src_inter)
    print_stats(f"{dataset} BEFORE", df)

    if args.verify_only:
        return

    new_df = build_temporal_split(
        df,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    print_stats(f"{out_dataset_name} AFTER", new_df)

    if args.output_style == "overwrite" and src_inter.exists():
        backup = src_inter.with_suffix(src_inter.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(src_inter, backup)
            print(f"[Backup] {backup}")
        else:
            print(f"[Backup exists] {backup}")

    if args.output_style == "new_dataset" and args.copy_side_files:
        copy_side_files(src_inter.parent, dst_dir, src_inter.name)

    dst_dir.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(dst_inter, sep="\t", index=False)
    print(f"[Saved] {dst_inter}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="leave_one_out",
        choices=["leave_one_out", "ratio"],
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument(
        "--output_style",
        type=str,
        default="new_dataset",
        choices=["new_dataset", "same_dir", "overwrite"],
        help=(
            "new_dataset: data/baby -> data/baby/baby.inter; "
            "same_dir: data/baby/baby.inter; "
            "overwrite: replace data/baby/baby.inter after creating .bak"
        ),
    )
    parser.add_argument(
        "--copy_side_files",
        action="store_true",
        help="When output_style=new_dataset, copy npy/csv/pt side files to the new dataset directory.",
    )
    parser.add_argument("--verify_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        process_dataset(args, dataset)


if __name__ == "__main__":
    main()
