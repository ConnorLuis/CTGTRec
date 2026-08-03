# CTGTRec final configurations

These files contain the fixed dataset-specific settings used for the reported
CTGTRec results.

## Formal configuration names

The released model uses method-oriented names rather than experiment-stage
labels:

| Configuration | Meaning |
| --- | --- |
| `trend_weight` | additive item-trend calibration coefficient |
| `trend_recent_ratio` | recent train-interaction proportion used for the global quantile |
| `trend_epsilon` | denominator and zero-variance tolerance |
| `trend_clip` | symmetric clipping bound after z-score normalization |
| `aux_loss_weight` | visual/textual auxiliary BPR weight |
| `visual_fusion_weight` | visual share of the frozen multimodal item graph |
| `ui_edge_dropout` | fraction of aggregated temporal user-item edges removed per epoch |
| `ct_raw_graph_file` | raw continuous-time weighted user-item graph |
| `ct_normalized_graph_file` | complete symmetric normalization used for evaluation |

## Item-trend calibration

The temporal interaction file is read from the active dataset configuration's
`inter_file_name`. Only rows with `x_label == 0` are used.

For recent ratio `r`, the threshold is the linear quantile `Q(1-r)` over all
training timestamps. Every interaction tied at that threshold is included. The
item signal is:

```text
all_rate_i    = all_count_i / number_of_train_interactions
recent_rate_i = recent_count_i / number_of_recent_interactions
relative_i    = recent_rate_i / (all_rate_i + trend_epsilon)
trend_i       = clip(zscore(log1p(relative_i)), -trend_clip, trend_clip)
```

Full-ranking prediction is:

```text
final_score(u, i) = graph_score(u, i) + trend_weight * trend_i
```

The trend term is not used in the BPR training loss.

## Dataset-specific parameters

| Dataset | LR | UI layers | tau | recent ratio | trend weight | auxiliary weight | edge dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby | 5e-4 | 5 | 0.30 | 0.45 | 1.10 | 1e-3 | 0.0 |
| Sports | 1e-3 | 6 | 0.10 | 0.45 | 1.10 | 1e-3 | 0.7 |
| Clothing | 5e-4 | 2 | 0.50 | 0.45 | 1.10 | 0.0 | 0.7 |
| MicroLens | 1e-3 | 1 | 0.03 | 0.25 | 1.10 | 0.0 | 0.0 |

## Multimodal graph cache

The frozen multimodal item graph is built from the original visual/textual
feature matrices, not from updated trainable feature embeddings. Its cache uses:

```text
mm_adj_ctgtrec_k<knn_k>_v<visual_fusion_weight>.pt
```

Caches created by earlier experimental implementations are not read by the
formal model and may be removed from local dataset directories.

## Three-seed evaluation

Each fixed configuration is repeated with seeds `999`, `2024`, and `3407`.
Each seed restores its best validation checkpoint and evaluates the test split
once. Report arithmetic mean and sample standard deviation (`ddof=1`); never
select a best seed.
