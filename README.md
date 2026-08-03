# CTGTRec

Official implementation of **CTGTRec: Continuous-Time Graph and Trend-aware Recommendation for Multimodal Recommendation**.

CTGTRec is a temporal multimodal recommendation model that combines:

1. a **continuous-time weighted user–item graph** for time-sensitive collaborative propagation;
2. a **frozen multimodal item–item graph** for relatively stable visual and textual semantics;
3. a **train-only item temporal trend** for candidate-level score calibration during full-ranking inference.

> This repository is being cleaned and organized for public release.  
> Exact preprocessing, training, and evaluation commands will be added after the source-code cleanup is completed.

## Overview

Most multimodal graph recommendation methods compress historical interactions into a static graph, while many temporal recommendation methods focus mainly on interaction sequences or ID-based dynamic states. CTGTRec jointly models temporal collaborative relations, multimodal item semantics, and candidate-item trend changes.

The model uses only training interactions to construct its graph structures and trend statistics.

### Continuous-Time User–Item Graph

For each user, historical interactions are assigned continuous edge weights according to their normalized temporal distance from the user's latest training interaction. An exponential decay function gives more recent interactions larger propagation weights.

The resulting weighted bipartite graph is symmetrically normalized and remains fixed during model training. Therefore, the “continuous-time graph” in CTGTRec is a timestamp-weighted training graph rather than an event-driven graph that updates node states online.

### Frozen Multimodal Item–Item Graph

Visual and textual item features are used to construct modality-specific k-nearest-neighbor graphs. The two graphs are normalized and fused into a fixed multimodal item–item graph.

This branch propagates relatively stable visual and textual semantic relations between items.

### Item Temporal Trend

CTGTRec estimates whether an item becomes more or less active near the end of the training period by comparing:

- its normalized frequency in a recent training window; and
- its normalized frequency over the complete training set.

The relative trend is transformed with `log1p`, standardized with a global z-score, and clipped to `[-3, 3]`.

This trend is computed entirely from training interactions and contains no learnable parameters.

### Trend-Aware Score Calibration

During full-ranking inference, the normalized item trend is added to the personalized recommendation score:

```text
final_score = personalized_score + trend_weight * normalized_item_trend
```

Trend calibration does not participate in graph propagation, the training loss, or backpropagation.

### Optimization

The model is trained with a BPR ranking objective based on the fused user and item representations. Optional visual and textual auxiliary BPR losses can provide additional modality-specific ranking supervision.

## Datasets

Experiments are conducted on four temporal multimodal recommendation datasets:

- **Baby**
- **Sports**
- **Clothing**
- **MicroLens**

Baby, Sports, and Clothing are Amazon product-review subsets. MicroLens is a short-video recommendation dataset.

The downloaded data include timestamped interactions and pre-extracted visual and textual item features. Dataset download and placement instructions are documented in `data/README.md`.

### Feature Dimensions

| Dataset | Visual dimension | Textual dimension |
| --- | ---: | ---: |
| Baby | 4096 | 384 |
| Sports | 4096 | 384 |
| Clothing | 4096 | 384 |
| MicroLens | 1024 | 1024 |

## Temporal Data Split

CTGTRec uses a strict per-user chronological split.

For every user:

1. interactions are sorted by `(timestamp, stable_index)`;
2. the last interaction is used for testing;
3. the second-to-last interaction is used for validation;
4. all earlier interactions are used for training.

The stable index is the original record order and is used as the secondary key when multiple interactions from the same user have identical timestamps.

Only training interactions are used to construct:

- the continuous-time user–item graph;
- item temporal trend statistics;
- model parameters.

Validation data are used for model selection, and test data are used only for final evaluation.

## Dataset Statistics

| Dataset | Users | Items | Interactions | Sparsity | Avg. train length | Median train length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby | 19,445 | 7,050 | 160,792 | 99.88% | 6.27 | 4 |
| Sports | 35,598 | 18,357 | 296,337 | 99.95% | 6.32 | 4 |
| Clothing | 39,387 | 23,033 | 278,677 | 99.97% | 5.08 | 4 |
| MicroLens | 98,129 | 17,228 | 705,174 | 99.96% | 5.19 | 4 |

## Environment

The paper experiments were conducted with the following environment:

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

The reported experiments used an NVIDIA GeForce RTX 4090. Other compatible GPUs can also be used, although runtime and memory consumption may differ.

## Installation

Create a Python 3.10 environment:

```bash
conda create -n ctgtrec python=3.10.20
conda activate ctgtrec
```

Install the CUDA 12.8 build of PyTorch and TorchVision:

```bash
pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` records the package versions used in the paper environment. PyTorch Geometric extension wheels must match the installed PyTorch and CUDA builds.

## Evaluation Protocol

The reported metrics are:

- Recall@10
- Recall@20
- NDCG@10
- NDCG@20

Model development primarily uses validation Recall@20. Final results are averaged over three random seeds:

```text
999, 2024, 3407
```

All models are evaluated with the same temporal split, candidate set, and full-ranking evaluator.

## Main Results

The following table reports the three-seed mean test performance of CTGTRec.

| Dataset | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| Baby | 0.0300 | 0.0501 | 0.0150 | 0.0201 |
| Sports | 0.0374 | 0.0592 | 0.0187 | 0.0241 |
| Clothing | 0.0335 | 0.0535 | 0.0168 | 0.0218 |
| MicroLens | 0.0552 | 0.0868 | 0.0272 | 0.0352 |

## Final Report Configurations

| Dataset | Learning rate | UI layers | Time scale | Recent ratio | Trend weight | Auxiliary weight | Dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby | 5e-4 | 5 | 0.30 | 0.45 | 1.1 | 1e-3 | 0.0 |
| Sports | 1e-3 | 6 | 0.10 | 0.45 | 1.1 | 1e-3 | 0.7 |
| Clothing | 5e-4 | 2 | 0.50 | 0.45 | 1.1 | 0 | 0.7 |
| MicroLens | 1e-3 | 1 | 0.03 | 0.25 | 1.1 | 0 | 0.0 |

Shared settings include:

- embedding dimension: `64`;
- multimodal k-nearest neighbors: `10`;
- visual graph fusion weight: `0.1`;
- multimodal propagation layers: `1`;
- optimizer: Adam;
- training batch size: `2048`;
- evaluation batch size: `512`;
- maximum epochs: `1000`;
- early-stopping patience: `20`;
- one negative sample per positive training interaction.

## Reproducibility Notes

- Graph structures and trend statistics must be built from training interactions only.
- Validation and test interactions must not be used to construct the user–item graph or item trend.
- Trend calibration is applied only during full-ranking inference.
- Interactions with identical timestamps must retain their original record order during the per-user split.
- The reported result for each dataset comes from one complete dataset-specific configuration rather than metric-wise hyperparameter selection.

## Citation

Citation information will be added after the publication metadata is finalized.

## Acknowledgements

This project uses the Amazon multimodal recommendation subsets and the MicroLens short-video recommendation dataset. Please follow the licenses and terms of use of the original datasets.
