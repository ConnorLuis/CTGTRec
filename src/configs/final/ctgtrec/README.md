# CTGTRec final configurations

These files contain the fixed dataset-specific configurations used for the
reported CTGTRec results. They are loaded automatically after `overall.yaml`,
the dataset YAML, and `model/CTGTRec.yaml`.

| Dataset | LR | UI layers | tau graph | recent ratio | trend weight | auxiliary weight | dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby | 5e-4 | 5 | 0.30 | 0.45 | 1.10 | 1e-3 | 0.0 |
| Sports | 1e-3 | 6 | 0.10 | 0.45 | 1.10 | 1e-3 | 0.7 |
| Clothing | 5e-4 | 2 | 0.50 | 0.45 | 1.10 | 0.0 | 0.7 |
| MicroLens | 1e-3 | 1 | 0.03 | 0.25 | 1.10 | 0.0 | 0.0 |

Shared settings remain in the standard configuration files: embedding size 64,
KNN `k=10`, visual fusion weight 0.1, one multimodal graph layer, Adam,
train/evaluation batch sizes 2048/512, 1000 epochs, early-stopping patience 20,
and one negative sample.

The three values in `overall.yaml` (`999`, `2024`, and `3407`) are repeated
random seeds, not candidate hyperparameters from which one result should be
selected. The run/aggregation code must report their mean and sample standard
deviation; that behavior is handled separately from these parameter files.

For Sports and Clothing, the released configuration records dropout `0.7`.
Correct use with a continuous-time weighted graph requires the model's dropout
implementation to retain the sampled temporal edge weights and renormalize the
sampled weighted graph. Do not treat the current legacy static-weight resampling
path as the final implementation.
