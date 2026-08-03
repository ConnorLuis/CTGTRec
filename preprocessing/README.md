# Data preprocessing

This directory contains the preprocessing utilities required to reproduce the
strict temporal data protocol and continuous-time user–item graph used by
CTGTRec.

Run all commands from the repository root.

## Preprocessing routes

The repository supports two starting points.

### Route A: use the released processed data

Use this route when the downloaded dataset package already contains re-indexed
interactions, ID mappings, and visual/textual features. Start from
[Build the strict temporal split](#1-build-the-strict-temporal-split).

### Route B: start from original raw data

Use the command-line scripts under [`preprocessing/raw/`](raw/) to construct
interactions, mappings, aligned Amazon metadata, and Amazon modality features.
See [`preprocessing/raw/README.md`](raw/README.md) for the complete raw-data
workflow and the original dataset sources.

After raw-data preprocessing, return to this document and run the same strict
temporal split and continuous-time graph construction steps as Route A.

## Required dataset layout

Before temporal splitting, each dataset directory should follow this layout:

```text
data/
├── baby/
│   ├── baby.inter
│   ├── image_feat.npy
│   ├── text_feat.npy
│   ├── i_id_mapping.csv
│   └── u_id_mapping.csv
├── sports/
│   ├── sports.inter
│   ├── image_feat.npy
│   ├── text_feat.npy
│   ├── i_id_mapping.csv
│   └── u_id_mapping.csv
├── clothing/
│   ├── clothing.inter
│   ├── image_feat.npy
│   ├── text_feat.npy
│   ├── i_id_mapping.csv
│   └── u_id_mapping.csv
└── microlens/
    ├── microlens.inter
    ├── image_feat.npy
    ├── text_feat.npy
    ├── i_id_mapping.csv
    └── u_id_mapping.csv
```

Each tab-separated interaction file must contain:

```text
userID    itemID    rating    timestamp    x_label
```

`userID` and `itemID` must be non-negative, zero-based, and contiguous. Feature
row `i` must correspond to `itemID == i`.

## 1. Build the strict temporal split

Run:

```bash
python preprocessing/build_temporal_split_inter.py \
  --data_root data \
  --datasets baby sports clothing microlens
```

The script reads:

```text
data/<dataset>/<dataset>.inter
```

and writes the derived file in the same dataset directory:

```text
data/<dataset>/<dataset>_temporal.inter
```

For each user, interactions are ordered by:

```text
(timestamp, original_record_order)
```

The original record order is used only to break exact timestamp ties. `itemID`
is not used as a tie-breaking key.

Labels are assigned as follows:

```text
all earlier interactions    -> train (x_label = 0)
second-to-last interaction  -> valid (x_label = 1)
last interaction            -> test  (x_label = 2)
```

CTGTRec uses 5-core datasets, so every user is expected to have at least three
interactions. The script raises an error rather than silently applying a
different protocol to shorter histories.

Only `x_label` is regenerated. User IDs, item IDs, ratings, timestamps, mapping
files, and feature-row alignment are not re-indexed or changed. Output rows are
written in per-user chronological order.

After this step, the expected layout is:

```text
data/
├── baby/
│   ├── baby.inter
│   ├── baby_temporal.inter
│   └── ...
├── sports/
│   ├── sports.inter
│   ├── sports_temporal.inter
│   └── ...
├── clothing/
│   ├── clothing.inter
│   ├── clothing_temporal.inter
│   └── ...
└── microlens/
    ├── microlens.inter
    ├── microlens_temporal.inter
    └── ...
```

To regenerate existing temporal files deliberately, add `--overwrite`:

```bash
python preprocessing/build_temporal_split_inter.py \
  --data_root data \
  --datasets baby sports clothing microlens \
  --overwrite
```

## 2. Build continuous-time user–item graphs

CTGTRec constructs each continuous-time graph from training interactions only:

```text
x_label == 0
```

For a training interaction `(u, i, t_ui)`, define:

```text
span_u   = t_last_u - t_first_u
delta_ui = (t_last_u - t_ui) / max(span_u, epsilon)
w_ui     = exp(-delta_ui / tau)
```

The timestamps used in `t_first_u`, `t_last_u`, and `span_u` are training
timestamps only. The resulting weighted undirected bipartite adjacency matrix is
symmetrically normalized:

```text
A_norm = D^(-1/2) A D^(-1/2)
```

If the same user and item have multiple training interactions, their temporal
weights are summed into the same matrix entry before normalization.

The final-report temporal scales are:

| Dataset | `tau` | Raw weighted graph | Normalized graph |
| --- | ---: | --- | --- |
| Baby | 0.30 | `ct_raw_adj_user_tau0p3.npz` | `ct_adj_user_tau0p3.npz` |
| Sports | 0.10 | `ct_raw_adj_user_tau0p1.npz` | `ct_adj_user_tau0p1.npz` |
| Clothing | 0.50 | `ct_raw_adj_user_tau0p5.npz` | `ct_adj_user_tau0p5.npz` |
| MicroLens | 0.03 | `ct_raw_adj_user_tau0p03.npz` | `ct_adj_user_tau0p03.npz` |

Build each final graph with its dataset-specific scale.

### Baby

```bash
python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets baby \
  --taus 0.30
```

### Sports

```bash
python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets sports \
  --taus 0.10
```

### Clothing

```bash
python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets clothing \
  --taus 0.50
```

### MicroLens

```bash
python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets microlens \
  --taus 0.03
```

The default input suffix is `_temporal.inter`. A custom suffix may be supplied
with `--input_suffix`, but the released CTGTRec pipeline uses the default.

To regenerate graph artifacts deliberately, add `--overwrite`.

## Graph outputs

For Baby, the generated directory is:

```text
data/baby/continuous_time_adj/
├── ct_raw_adj_user_tau0p3.npz
├── ct_adj_user_tau0p3.npz
├── ct_adj_stats.csv
└── ct_adj_manifest.json
```

The other datasets use the same directory and filename pattern.

The `ct_raw_adj_user_tau*.npz` file stores unnormalized aggregated temporal
weights and is required for weighted edge dropout. The `ct_adj_user_tau*.npz`
file stores the complete symmetric normalization used for evaluation.

`ct_adj_stats.csv` records graph and temporal-weight statistics.
`ct_adj_manifest.json` records the source interaction file, train/validation/test
counts, graph formula, `epsilon`, zero-span user count, duplicate user–item
policy, feature/mapping checks, and generated graph paths.

For diagnostics, add:

```text
--save_edge_values
```

This additionally writes `ct_edge_values.csv`, containing the training edge
timestamps, user-normalized temporal distances, and temporal weights. This file
can be large and is not required for training.

The final script does not generate global-time graphs, static comparison graphs,
timestamp snapshots, or minimum-weight-clipped graphs.

## Train-only and evaluation rules

The following artifacts must be derived from training interactions only:

- continuous-time user–item graph edges and temporal statistics;
- item temporal trend statistics;
- model parameters.

Validation interactions are used for model selection. Test interactions are
used only for final evaluation. Validation and test interactions must not be
used to construct graph edges, graph time ranges, or item trend statistics.

## Validation and failure behavior

The preprocessing scripts fail explicitly when they detect conditions that can
break reproducibility or feature alignment, including:

- missing or malformed interaction columns;
- non-integer or negative user/item IDs;
- non-contiguous zero-based IDs;
- missing train/validation/test labels;
- non-finite timestamps;
- mapping counts inconsistent with interaction IDs;
- feature row counts inconsistent with the number of items;
- existing output artifacts without `--overwrite`;
- non-positive `tau` or `epsilon`;
- non-symmetric or non-finite generated adjacency matrices.

Do not bypass these checks by editing generated files manually.

## Raw-data scripts

The public raw-data directory contains:

```text
preprocessing/raw/
├── README.md
├── build_interactions.py
├── split_interactions.py
├── reindex_amazon_metadata.py
└── encode_amazon_features.py
```

These scripts replace the legacy notebooks. The canonical split implementation
remains `preprocessing/build_temporal_split_inter.py`; the raw-data
`split_interactions.py` entry point delegates to that implementation rather than
maintaining a separate split algorithm.

## Generated artifacts

Dataset contents and generated graph artifacts should not be committed to Git.
The repository ignore rules should cover at least:

```gitignore
data/*
!data/README.md
**/continuous_time_adj/
*.bak
*.tmp
.ipynb_checkpoints/
__pycache__/
```

## Reproduction checklist

Before training CTGTRec, verify that:

1. `data/<dataset>/<dataset>_temporal.inter` exists;
2. every user has exactly one validation and one test interaction;
3. IDs are contiguous and feature rows align with `itemID`;
4. both dataset-specific raw and normalized `tau` graph files exist;
5. the graph manifest reports `graph_source` as training interactions only;
6. no validation or test interaction was used to build graph or trend features.
