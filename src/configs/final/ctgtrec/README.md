# CTGTRec final configurations

These files contain the fixed dataset-specific configurations used for the
reported CTGTRec results.

## Three-seed protocol

The final configuration is repeated with the fixed seeds:

```text
999, 2024, 3407
```

For each seed:

1. initialize a fresh model and optimizer;
2. train with validation-only early stopping;
3. snapshot the checkpoint whenever `Recall@20` improves;
4. restore that seed's best validation checkpoint;
5. evaluate the test split exactly once.

The three seeds are not a hyperparameter search and no "best seed" is selected.
For each metric, the public runner reports:

```text
arithmetic mean
sample standard deviation with ddof = 1
```

The standard reported metrics are `Recall@10`, `Recall@20`, `NDCG@10`, and
`NDCG@20`. Per-seed and aggregate outputs are written under:

```text
results/ctgtrec/<dataset>/combo_000/
├── seed_results.csv
├── summary.csv
└── summary.json
```

Checkpoints are written under `saved/` when model saving is enabled.

## Dataset-specific parameters

| Dataset | LR | UI layers | tau | recent ratio | trend weight | auxiliary weight | dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby | 5e-4 | 5 | 0.30 | 0.45 | 1.10 | 1e-3 | 0.0 |
| Sports | 1e-3 | 6 | 0.10 | 0.45 | 1.10 | 1e-3 | 0.7 |
| Clothing | 5e-4 | 2 | 0.50 | 0.45 | 1.10 | 0.0 | 0.7 |
| MicroLens | 1e-3 | 1 | 0.03 | 0.25 | 1.10 | 0.0 | 0.0 |

Each dataset explicitly names both continuous-time graph artifacts:

- `ct_raw_adj_user_tau*.npz`: unnormalized temporal weights used for weighted
  edge dropout;
- `ct_adj_user_tau*.npz`: complete symmetric normalization used for evaluation.
