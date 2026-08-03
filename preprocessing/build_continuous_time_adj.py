#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_continuous_time_adj.py

Build continuous-time weighted user-item adjacency matrices from *_temporal.inter files.

Input:
  data/<dataset>/<inter_file> or data/<dataset>_temporal/<dataset>_temporal.inter

Output:
  data/<dataset>/<output_dir>/ct_adj_static.npz
  data/<dataset>/<output_dir>/ct_adj_<mode>_tau<tau>.npz
  data/<dataset>/<output_dir>/ct_adj_manifest.json
  data/<dataset>/<output_dir>/ct_adj_stats.csv

Weight definition:
  Each training interaction is an edge (userID, itemID, timestamp).
  We do NOT discretize timestamps into snapshots. Instead, timestamp is used as an edge attribute.

  global mode:
      delta = (global_train_t_max - t) / global_train_span

  user mode:
      delta = (user_train_t_max[u] - t) / user_train_span[u]
      if a user's train span is 0, fall back to global_train_span.

  continuous-time kernel:
      w = exp(-delta / tau)

  Then weighted UI graph is symmetrically normalized:
      A_norm = D^{-1/2} A D^{-1/2}

Usage examples:
  # If files are data/baby/baby.inter, data/sports/sports.inter, ...
  python preprocessing/build_continuous_time_adj.py \
      --data_root data \
      --datasets baby sports clothing microlens \
      --inter_suffix _temporal.inter \
      --modes user global \
      --taus 0.1 0.3 1.0 \
      --output_dir continuous_time_adj

  # If files are data/baby/baby.inter, ...
  python preprocessing/build_continuous_time_adj.py \
      --data_root data \
      --datasets baby sports clothing microlens \
      --inter_suffix .inter \
      --modes user global \
      --taus 0.1 0.3 1.0 \
      --output_dir continuous_time_adj
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


REQUIRED_COLS = ["userID", "itemID", "timestamp"]


def _safe_float_for_name(x: float) -> str:
    """Convert 0.1 -> 0p1 for safe filenames."""
    if math.isinf(x):
        return "inf"
    s = ("%.12g" % float(x)).replace("-", "m").replace(".", "p")
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Support both plain MMRec columns and typed columns like userID:token."""
    rename = {}
    for c in df.columns:
        base = str(c).split(":", 1)[0]
        rename[c] = base
    return df.rename(columns=rename)


def _read_inter(path: Path, sep: str = "\t") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"inter file not found: {path}")
    df = pd.read_csv(path, sep=sep)
    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}; got columns={list(df.columns)}")
    if "x_label" not in df.columns:
        raise ValueError(f"{path} missing x_label column. This script expects temporal split labels.")
    return df


def _find_inter_path(data_root: Path, dataset: str, inter_suffix: Optional[str]) -> Tuple[Path, Path]:
    """Return (dataset_dir, inter_path) with several fallbacks."""
    dataset_dir = data_root / dataset

    candidates: List[Path] = []
    if inter_suffix:
        # inter_suffix can be "_temporal.inter" for dataset=baby -> baby.inter,
        # or ".inter" for dataset=baby -> baby.inter.
        if inter_suffix.startswith(".") or inter_suffix.startswith("_"):
            candidates.append(dataset_dir / f"{dataset}{inter_suffix}")
        else:
            candidates.append(dataset_dir / inter_suffix)

    candidates.extend([
        dataset_dir / f"{dataset}.inter",
        dataset_dir / f"{dataset}_temporal.inter",
    ])

    # If dataset=baby, also try data/baby/baby.inter.
    if not dataset.endswith("_temporal"):
        temporal_dataset = f"{dataset}_temporal"
        candidates.extend([
            data_root / temporal_dataset / f"{temporal_dataset}.inter",
            data_root / temporal_dataset / f"{dataset}_temporal.inter",
        ])

    for p in candidates:
        if p.exists():
            return p.parent, p

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find inter file for dataset={dataset}. Searched:\n  {searched}")


def _infer_num_nodes(df_all: pd.DataFrame, dataset_dir: Path) -> Tuple[int, int]:
    """Infer n_users and n_items. Prefer mapping/feature files when available."""
    n_users = int(df_all["userID"].max()) + 1
    n_items = int(df_all["itemID"].max()) + 1

    # Mapping files can confirm counts when present.
    u_map = dataset_dir / "u_id_mapping.csv"
    i_map = dataset_dir / "i_id_mapping.csv"
    if u_map.exists():
        try:
            u_df = pd.read_csv(u_map)
            if "userID" in u_df.columns:
                n_users = max(n_users, int(u_df["userID"].max()) + 1)
            else:
                n_users = max(n_users, len(u_df))
        except Exception:
            pass
    if i_map.exists():
        try:
            i_df = pd.read_csv(i_map)
            if "itemID" in i_df.columns:
                n_items = max(n_items, int(i_df["itemID"].max()) + 1)
            else:
                n_items = max(n_items, len(i_df))
        except Exception:
            pass

    # Feature files are row-aligned by itemID in MMRec.
    for feat_name in ["image_feat.npy", "text_feat.npy"]:
        feat_path = dataset_dir / feat_name
        if feat_path.exists():
            try:
                arr = np.load(feat_path, mmap_mode="r")
                n_items = max(n_items, int(arr.shape[0]))
            except Exception:
                pass
    return n_users, n_items


def _aggregate_edges(
    users: np.ndarray,
    items: np.ndarray,
    weights: np.ndarray,
    n_users: int,
    n_items: int,
) -> sp.csr_matrix:
    """Build weighted bipartite adjacency and aggregate duplicate user-item edges."""
    users = users.astype(np.int64, copy=False)
    items = items.astype(np.int64, copy=False)
    weights = weights.astype(np.float32, copy=False)

    row = np.concatenate([users, n_users + items])
    col = np.concatenate([n_users + items, users])
    data = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    n_nodes = n_users + n_items

    adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)
    adj.sum_duplicates()
    return adj.tocsr()


def _symmetric_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    rowsum = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float64)
    inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    nonzero = rowsum > 0
    inv_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)
    d_mat = sp.diags(inv_sqrt.astype(np.float32))
    norm = d_mat @ adj @ d_mat
    return norm.tocsr().astype(np.float32)


def _compute_delta(
    train_df: pd.DataFrame,
    mode: str,
    global_span_eps: float = 1.0,
) -> np.ndarray:
    ts = train_df["timestamp"].to_numpy(dtype=np.float64)
    t_min = float(np.min(ts))
    t_max = float(np.max(ts))
    global_span = max(t_max - t_min, global_span_eps)

    if mode == "global":
        delta = (t_max - ts) / global_span
        return np.maximum(delta, 0.0)

    if mode == "user":
        tmp = train_df[["userID", "timestamp"]].copy()
        grouped = tmp.groupby("userID")["timestamp"]
        u_max = grouped.transform("max").to_numpy(dtype=np.float64)
        u_min = grouped.transform("min").to_numpy(dtype=np.float64)
        u_span = u_max - u_min
        # Users with only one training edge get delta=0 for that edge.
        denom = np.where(u_span > 0, u_span, global_span)
        delta = (u_max - ts) / denom
        return np.maximum(delta, 0.0)

    raise ValueError(f"Unsupported mode={mode}. Choose from: global, user")


def _make_weights(delta: np.ndarray, tau: float, min_weight: float = 0.0) -> np.ndarray:
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    weights = np.exp(-delta / float(tau)).astype(np.float32)
    if min_weight > 0:
        weights = np.maximum(weights, np.float32(min_weight))
    return weights


def _save_graph(
    out_path: Path,
    users: np.ndarray,
    items: np.ndarray,
    weights: np.ndarray,
    n_users: int,
    n_items: int,
) -> Dict[str, float]:
    raw_adj = _aggregate_edges(users, items, weights, n_users, n_items)
    norm_adj = _symmetric_normalize(raw_adj)
    sp.save_npz(out_path, norm_adj)

    raw_degree = np.asarray(raw_adj.sum(axis=1)).reshape(-1)
    stats = {
        "path": str(out_path),
        "raw_edge_count": int(len(weights)),
        "raw_adj_nnz": int(raw_adj.nnz),
        "norm_adj_nnz": int(norm_adj.nnz),
        "weight_min": float(np.min(weights)) if len(weights) else 0.0,
        "weight_max": float(np.max(weights)) if len(weights) else 0.0,
        "weight_mean": float(np.mean(weights)) if len(weights) else 0.0,
        "weight_std": float(np.std(weights)) if len(weights) else 0.0,
        "zero_degree_nodes": int(np.sum(raw_degree == 0)),
    }
    return stats


def build_for_dataset(
    data_root: Path,
    dataset: str,
    inter_suffix: Optional[str],
    output_dir_name: str,
    modes: Iterable[str],
    taus: Iterable[float],
    min_weight: float,
    sep: str,
    save_edge_weights: bool,
) -> Dict:
    dataset_dir, inter_path = _find_inter_path(data_root, dataset, inter_suffix)
    df_all = _read_inter(inter_path, sep=sep)

    n_users, n_items = _infer_num_nodes(df_all, dataset_dir)
    train_df = df_all[df_all["x_label"].astype(int) == 0].copy()
    if train_df.empty:
        raise ValueError(f"No training interactions x_label=0 in {inter_path}")

    train_df["userID"] = train_df["userID"].astype(np.int64)
    train_df["itemID"] = train_df["itemID"].astype(np.int64)
    train_df["timestamp"] = train_df["timestamp"].astype(np.float64)

    users = train_df["userID"].to_numpy(dtype=np.int64)
    items = train_df["itemID"].to_numpy(dtype=np.int64)
    ts = train_df["timestamp"].to_numpy(dtype=np.float64)

    if users.max(initial=0) >= n_users or items.max(initial=0) >= n_items:
        raise ValueError(
            f"ID out of range in {inter_path}: max_user={users.max()}, n_users={n_users}, "
            f"max_item={items.max()}, n_items={n_items}"
        )

    out_dir = dataset_dir / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_rows: List[Dict] = []

    # Static sanity graph: all training edges have weight=1.
    static_path = out_dir / "ct_adj_static.npz"
    static_weights = np.ones(len(train_df), dtype=np.float32)
    static_stats = _save_graph(static_path, users, items, static_weights, n_users, n_items)
    static_stats.update({"dataset": dataset, "mode": "static", "tau": "inf"})
    stats_rows.append(static_stats)

    manifest = {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "inter_path": str(inter_path),
        "output_dir": str(out_dir),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_nodes": int(n_users + n_items),
        "num_all_interactions": int(len(df_all)),
        "num_train_interactions": int(len(train_df)),
        "train_timestamp_min": float(np.min(ts)),
        "train_timestamp_max": float(np.max(ts)),
        "train_timestamp_span": float(np.max(ts) - np.min(ts)),
        "formula": "w = exp(-delta / tau); A_norm = D^-1/2 A D^-1/2",
        "graphs": [static_stats],
    }

    for mode in modes:
        delta = _compute_delta(train_df, mode=mode)
        if save_edge_weights:
            edge_df = train_df[["userID", "itemID", "timestamp"]].copy()
            edge_df[f"delta_{mode}"] = delta
            edge_df.to_csv(out_dir / f"ct_edge_delta_{mode}.csv", index=False)

        for tau in taus:
            weights = _make_weights(delta, tau=tau, min_weight=min_weight)
            tau_name = _safe_float_for_name(tau)
            out_path = out_dir / f"ct_adj_{mode}_tau{tau_name}.npz"
            graph_stats = _save_graph(out_path, users, items, weights, n_users, n_items)
            graph_stats.update({"dataset": dataset, "mode": mode, "tau": float(tau)})
            stats_rows.append(graph_stats)
            manifest["graphs"].append(graph_stats)

            if save_edge_weights:
                edge_df = train_df[["userID", "itemID", "timestamp"]].copy()
                edge_df["weight"] = weights
                edge_df.to_csv(out_dir / f"ct_edge_weight_{mode}_tau{tau_name}.csv", index=False)

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = out_dir / "ct_adj_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    manifest_path = out_dir / "ct_adj_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] dataset={dataset}")
    print(f"  inter: {inter_path}")
    print(f"  output: {out_dir}")
    print(f"  n_users={n_users}, n_items={n_items}, train_edges={len(train_df)}")
    print(stats_df[["mode", "tau", "raw_edge_count", "norm_adj_nnz", "weight_min", "weight_max", "weight_mean"]].to_string(index=False))

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build continuous-time weighted UI adjacency matrices from temporal .inter files.")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory, e.g., data or ../data")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Dataset names, e.g., baby sports clothing microlens or baby ...")
    parser.add_argument("--inter_suffix", type=str, default=None, help="Optional suffix/template. Examples: _temporal.inter, .inter")
    parser.add_argument("--output_dir", type=str, default="continuous_time_adj", help="Output directory name under each dataset directory")
    parser.add_argument("--modes", type=str, nargs="+", default=["user", "global"], choices=["user", "global"], help="Time delta modes")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.3, 1.0], help="Positive temporal kernel scales")
    parser.add_argument("--min_weight", type=float, default=0.0, help="Optional lower bound for edge weights")
    parser.add_argument("--sep", type=str, default="\t", help="Column separator for .inter files")
    parser.add_argument("--save_edge_weights", action="store_true", help="Also save per-edge deltas/weights as CSV; can be large")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    all_manifests = []
    for dataset in args.datasets:
        manifest = build_for_dataset(
            data_root=data_root,
            dataset=dataset,
            inter_suffix=args.inter_suffix,
            output_dir_name=args.output_dir,
            modes=args.modes,
            taus=args.taus,
            min_weight=args.min_weight,
            sep=args.sep,
            save_edge_weights=args.save_edge_weights,
        )
        all_manifests.append(manifest)

    print("\n[DONE] Built continuous-time adjacency matrices for all datasets.")


if __name__ == "__main__":
    main()
