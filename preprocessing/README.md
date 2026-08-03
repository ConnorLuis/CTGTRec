# Data Preprocessing

This directory contains the preprocessing utilities required to reproduce the
temporal data protocol and continuous-time user–item graph used by CTGTRec.

The public dataset package already provides re-indexed interactions, ID mapping
files, and pre-extracted visual/textual features. Therefore, the legacy raw-data
notebooks are not required for normal reproduction.

## Required input files

After downloading the datasets, place them under `data/`:

```text
data/
├── baby/
│   ├── baby.inter
│   ├── image_feat.npy
│   ├── text_feat.npy
│   ├── i_id_mapping.csv
│   └── u_id_mapping.csv
├── sports/
├── clothing/
└── microlens/
```

Each interaction file must contain the following tab-separated columns:

```text
userID    itemID    rating    timestamp    x_label
```

The existing `x_label` values are replaced by the paper's temporal split.

## Step 1: Build the strict per-user temporal split

Run:

```bash
python preprocessing/build_temporal_split_inter.py \
  --data_root data \
  --datasets baby sports clothing microlens \
  --output_style new_dataset \
  --copy_side_files
```

For each user, interactions are stably ordered by:

```text
(timestamp, original_record_order)
```

The split is:

```text
last interaction          -> test  (x_label = 2)
second-to-last interaction -> valid (x_label = 1)
all earlier interactions   -> train (x_label = 0)
```

The command creates:

```text
data/
├── baby_temporal/
│   ├── baby_temporal.inter
│   ├── image_feat.npy
│   ├── text_feat.npy
│   ├── i_id_mapping.csv
│   └── u_id_mapping.csv
├── sports_temporal/
├── clothing_temporal/
└── microlens_temporal/
```

Only the `x_label` column is regenerated. User IDs, item IDs, ratings,
timestamps, mappings, and feature-row alignment remain unchanged.

## Step 2: Build continuous-time user–item adjacency matrices

CTGTRec uses only training interactions (`x_label = 0`) to construct the
continuous-time weighted user–item graph.

For an interaction `(u, i, t)`, the user-normalized temporal distance is:

```text
delta_ui = (t_last_u - t_ui) / max(t_last_u - t_first_u, epsilon)
```

The continuous edge weight is:

```text
w_ui = exp(-delta_ui / tau)
```

The weighted bipartite adjacency matrix is then symmetrically normalized.

Build the final-report graph for each dataset:

```bash
python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets baby_temporal \
  --inter_suffix .inter \
  --modes user \
  --taus 0.30

python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets sports_temporal \
  --inter_suffix .inter \
  --modes user \
  --taus 0.10

python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets clothing_temporal \
  --inter_suffix .inter \
  --modes user \
  --taus 0.50

python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets microlens_temporal \
  --inter_suffix .inter \
  --modes user \
  --taus 0.03
```

Expected graph files:

```text
data/baby_temporal/continuous_time_adj/ct_adj_user_tau0p3.npz
data/sports_temporal/continuous_time_adj/ct_adj_user_tau0p1.npz
data/clothing_temporal/continuous_time_adj/ct_adj_user_tau0p5.npz
data/microlens_temporal/continuous_time_adj/ct_adj_user_tau0p03.npz
```

The script also writes a manifest and graph statistics for reproducibility.

## Train-only rule

The following artifacts must be derived from training interactions only:

- the continuous-time user–item graph;
- item temporal trend statistics;
- model parameters.

Validation and test interactions must not be used to build graph structures or
trend statistics.

## Files intentionally excluded

The public release does not include legacy notebooks for raw Amazon processing,
random/ratio splitting, static-time user–item graph fusion, or exploratory
temporal diagnostics. Those files are not part of the final CTGTRec method and
can cause the released code to deviate from the paper protocol.

## Generated files

Dataset files and generated adjacency matrices should not be committed to Git.
Add the following entries to the repository `.gitignore`:

```gitignore
data/*
!data/README.md
**/continuous_time_adj/
*.bak
.ipynb_checkpoints/
__pycache__/
```
