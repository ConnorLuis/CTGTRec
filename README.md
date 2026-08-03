# CTGTRec

[![CI](https://github.com/ConnorLuis/CTGTRec/actions/workflows/ci.yml/badge.svg)](https://github.com/ConnorLuis/CTGTRec/actions/workflows/ci.yml)

Official research implementation of **CTGTRec: Continuous-Time Graph Learning and Item-Trend-Aware Score Calibration for Multimodal Recommendation**.

CTGTRec combines:

1. a **continuous-time weighted user-item graph** for time-sensitive collaborative propagation;
2. a **frozen multimodal item-item graph** for visual and textual semantic propagation;
3. a **train-only item temporal trend** added during full-ranking inference.

> **Publication status.** The manuscript title, venue, DOI, affiliation, and final
> bibliographic metadata have not yet been finalized. The `Citation` section
> therefore provides a temporary manuscript citation that should be updated
> after publication.

## Method at a Glance

```text
Timestamped train interactions
        │
        ├── per-user continuous-time weighting
        │          ↓
        │   weighted user-item graph
        │          ↓
        │   collaborative propagation
        │
Visual/textual item features
        │
        ├── modality-specific kNN graphs
        │          ↓
        │   frozen fused item-item graph
        │          ↓
        │   semantic propagation
        │
Train timestamps and item frequencies
        │
        └── recent-vs-all relative activity
                   ↓
             log1p → z-score → clip
                   ↓
      additive item-trend score calibration
```

The continuous-time graph is constructed once from training interactions and
its temporal edge weights are not learned or updated from validation/test data.
For the Sports and Clothing final configurations, training-only weighted edge
dropout samples from this fixed graph as a regularizer. Validation and test
prediction always use the complete normalized graph.

The item-trend term is non-parametric and is used only at full-ranking
inference:

```text
final_score(u, i)
    = personalized_graph_score(u, i)
    + trend_weight × normalized_item_trend(i)
```

It does not participate in graph propagation, the BPR loss, or backpropagation.

## Reproducibility Guarantees

The public pipeline enforces the following protocol:

- interactions are split per user by `(timestamp, stable original record order)`;
- the final interaction is test, the second-to-last is validation, and the rest
  are training;
- only training interactions are used to build the continuous-time graph and
  item-trend statistics;
- model selection uses validation Recall@20;
- each fixed configuration runs seeds `999`, `2024`, and `3407`;
- no "best seed" is selected;
- each seed restores its best validation checkpoint before one final test
  evaluation;
- final metrics are reported as arithmetic mean and sample standard deviation
  (`ddof=1`);
- the standard metrics are Recall@10, Recall@20, NDCG@10, and NDCG@20.

## Main Results

Three-seed mean test results reported in the manuscript:

| Dataset | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| Baby | 0.0300 | 0.0501 | 0.0150 | 0.0201 |
| Sports | 0.0374 | 0.0592 | 0.0187 | 0.0241 |
| Clothing | 0.0335 | 0.0535 | 0.0168 | 0.0218 |
| MicroLens | 0.0552 | 0.0868 | 0.0272 | 0.0352 |

The repository writes full-precision per-seed results and aggregate mean/sample
standard deviation files. The main table above shows the rounded means used in
the manuscript.

## Repository Structure

```text
CTGTRec/
├── README.md
├── LICENSE
├── NOTICE
├── THIRD_PARTY_NOTICES.md
├── CITATION.cff
├── requirements.txt
├── requirements-preprocessing.txt
│
├── data/
│   └── README.md
│
├── docs/
│   └── RUNNING.md
│
├── preprocessing/
│   ├── README.md
│   ├── build_temporal_split_inter.py
│   ├── build_continuous_time_adj.py
│   └── raw/
│       ├── README.md
│       ├── build_interactions.py
│       ├── split_interactions.py
│       ├── reindex_amazon_metadata.py
│       └── encode_amazon_features.py
│
└── src/
    ├── main.py
    ├── common/
    │   └── trainer.py
    ├── configs/
    │   ├── overall.yaml
    │   ├── dataset/
    │   ├── model/
    │   └── final/ctgtrec/
    ├── models/
    │   ├── ctgtrec.py
    │   └── baseline implementations
    └── utils/
        ├── configurator.py
        ├── quick_start.py
        └── topk_evaluator.py
```

Key documentation:

- [Environment, commands, and output files](docs/RUNNING.md)
- [Automated tests and continuous integration](docs/TESTING.md)
- [Dataset download and placement](data/README.md)
- [Temporal split and continuous-time graph preprocessing](preprocessing/README.md)
- [Raw Amazon preprocessing](preprocessing/raw/README.md)
- [Final CTGTRec configurations](src/configs/final/ctgtrec/README.md)

## Environment

Reference experiment environment:

| Component | Version |
| --- | --- |
| Operating system | Ubuntu 22.04.3 LTS |
| Python | 3.10.20 |
| PyTorch | 2.9.1 with CUDA 12.8 |
| TorchVision | 0.24.1 |
| PyTorch Geometric | 2.8.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| PyYAML | 6.0.3 |
| GPU used for reported experiments | NVIDIA GeForce RTX 4090 |

Other compatible hardware may be used, but runtime, memory use, and low-level
floating-point behavior can differ.

## Installation

Create and activate a Python 3.10 environment:

```bash
conda create -n ctgtrec python=3.10.20
conda activate ctgtrec
```

Install the CUDA 12.8 PyTorch build first:

```bash
pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining runtime dependencies:

```bash
pip install -r requirements.txt
```

Raw Amazon text encoding additionally requires:

```bash
pip install -r requirements-preprocessing.txt
```

See [docs/RUNNING.md](docs/RUNNING.md) for CPU usage, GPU selection, debugging
overrides, and expected output paths.

## Data Preparation

CTGTRec is evaluated on:

- Baby
- Sports
- Clothing
- MicroLens

Baby, Sports, and Clothing are Amazon product-review subsets. MicroLens is a
short-video recommendation dataset. This repository does not redistribute the
original datasets.

Place each dataset under:

```text
data/<dataset>/
├── <dataset>_temporal.inter
├── image_feat.npy
├── text_feat.npy
├── i_id_mapping.csv
├── u_id_mapping.csv
└── continuous_time_adj/
    ├── ct_raw_adj_user_tau*.npz
    └── ct_adj_user_tau*.npz
```

Download and placement instructions are in [data/README.md](data/README.md).

### Strict Temporal Split

From the repository root:

```bash
python preprocessing/build_temporal_split_inter.py \
  --data_root data \
  --datasets baby sports clothing microlens
```

For each user, the split script sorts by:

```text
(timestamp, original_record_order)
```

and assigns:

```text
last interaction          → test
second-to-last interaction → validation
all earlier interactions   → train
```

### Continuous-Time Graphs

Generate the final graph required by each dataset:

```bash
python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets baby \
  --taus 0.30 \
  --overwrite

python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets sports \
  --taus 0.10 \
  --overwrite

python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets clothing \
  --taus 0.50 \
  --overwrite

python preprocessing/build_continuous_time_adj.py \
  --data_root data \
  --datasets microlens \
  --taus 0.03 \
  --overwrite
```

For each time scale, preprocessing writes:

```text
ct_raw_adj_user_tau*.npz  # raw aggregated temporal weights
ct_adj_user_tau*.npz      # complete symmetric normalization
ct_adj_stats.csv
ct_adj_manifest.json
```

The raw graph is required for weighted training-time edge dropout. The complete
normalized graph is used for no-dropout training and full-ranking evaluation.

## Quick Start

All commands are run from the repository root.

Check the resolved configuration without loading data:

```bash
python src/main.py \
  --model CTGTRec \
  --dataset baby \
  --show-config
```

Run the four final three-seed experiments:

```bash
python src/main.py --model CTGTRec --dataset baby
python src/main.py --model CTGTRec --dataset sports
python src/main.py --model CTGTRec --dataset clothing
python src/main.py --model CTGTRec --dataset microlens
```

Select a GPU:

```bash
python src/main.py --model CTGTRec --dataset baby --gpu-id 1
```

Force CPU execution:

```bash
python src/main.py --model CTGTRec --dataset baby --cpu
```

Temporary debugging overrides are supported:

```bash
python src/main.py \
  --model CTGTRec \
  --dataset baby \
  --set epochs=2 \
  --set stopping_step=1
```

Do not use debugging overrides when reproducing the fixed paper configurations.

## Final CTGTRec Configurations

| Dataset | Learning rate | UI layers | Time scale | Recent ratio | Trend weight | Auxiliary weight | UI edge dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby | 5e-4 | 5 | 0.30 | 0.45 | 1.10 | 1e-3 | 0.0 |
| Sports | 1e-3 | 6 | 0.10 | 0.45 | 1.10 | 1e-3 | 0.7 |
| Clothing | 5e-4 | 2 | 0.50 | 0.45 | 1.10 | 0.0 | 0.7 |
| MicroLens | 1e-3 | 1 | 0.03 | 0.25 | 1.10 | 0.0 | 0.0 |

Shared settings:

| Setting | Value |
| --- | ---: |
| Embedding dimension | 64 |
| Multimodal kNN neighbors | 10 |
| Visual graph fusion weight | 0.1 |
| Multimodal propagation layers | 1 |
| Optimizer | Adam |
| Training batch size | 2048 |
| Evaluation batch size | 512 |
| Maximum epochs | 1000 |
| Early-stopping patience | 20 |
| Negative samples per positive | 1 |

The authoritative files are under:

```text
src/configs/model/CTGTRec.yaml
src/configs/final/ctgtrec/
```

## Outputs

For a fixed CTGTRec dataset configuration:

```text
results/ctgtrec/<dataset>/combo_000/
├── seed_results.csv
├── summary.csv
└── summary.json
```

When checkpoint saving is enabled:

```text
saved/
├── ctgtrec-<dataset>-combo000-seed999.pth
├── ctgtrec-<dataset>-combo000-seed2024.pth
└── ctgtrec-<dataset>-combo000-seed3407.pth
```

Logs are written under `logs/`.

## Included Comparison Models

The manuscript compares CTGTRec with 15 external baselines:

```text
BPR-MF, LightGCN,
VBPR, MMGCN, GRCN, LATTICE, BM3, SLMRec, MGCN, FREEDOM,
MISSRec, HM4SR, M3Rec, MuSTRec,
TimeMM
```

These cover collaborative filtering, static graph recommendation, multimodal
graph recommendation, multimodal sequential recommendation, and dynamic
multimodal recommendation. Baseline implementations and adapters should be
interpreted together with their source comments, model YAML files, cited
papers, and any upstream repositories listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

The final venue and DOI are not yet available. Until the publication metadata
is finalized, cite the manuscript and software as follows:

```bibtex
@misc{lv2026ctgtrec,
  author       = {Kangnan Lv},
  title        = {CTGTRec: Continuous-Time Graph Learning and
                  Item-Trend-Aware Score Calibration for
                  Multimodal Recommendation},
  year         = {2026},
  note         = {Manuscript},
  howpublished = {\url{https://github.com/ConnorLuis/CTGTRec}}
}
```

GitHub citation metadata is also provided in [CITATION.cff](CITATION.cff).
Replace the temporary entry with the final venue, volume/pages, DOI, and author
metadata after publication.

## License and Provenance

This repository is released under the **GNU General Public License version 3
only** (`GPL-3.0-only`). See [LICENSE](LICENSE).

The codebase is built on and substantially adapted from the
[MMRec](https://github.com/enoche/MMRec) multimodal recommendation toolbox,
which is distributed under GPLv3. CTGTRec-specific model code, temporal
preprocessing, configuration handling, evaluation protocol, documentation, and
other modifications were added in 2026.

See:

- [NOTICE](NOTICE) for the prominent modification notice;
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream software and
  dataset provenance.

Dataset files and pretrained feature artifacts are not covered merely by this
repository's software license. Their original licenses and terms of use remain
applicable.

## Acknowledgements

We thank the MMRec contributors for the base recommendation framework and
baseline implementations. We also thank the maintainers of the Amazon Review
Data and MicroLens datasets and the authors of all comparison methods.
