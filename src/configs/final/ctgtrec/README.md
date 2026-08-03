# CTGTRec final configurations

These files contain the fixed dataset-specific configurations used for the
reported CTGTRec results. Each dataset explicitly names both graph artifacts:

- `ct_raw_adj_user_tau*.npz`: unnormalized continuous-time weights, used only to
  sample and renormalize the weighted training graph when dropout is enabled;
- `ct_adj_user_tau*.npz`: the complete symmetrically normalized graph, used for
  no-dropout training and full-ranking evaluation.

| Dataset | LR | UI layers | tau | recent ratio | trend weight | auxiliary weight | dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby | 5e-4 | 5 | 0.30 | 0.45 | 1.10 | 1e-3 | 0.0 |
| Sports | 1e-3 | 6 | 0.10 | 0.45 | 1.10 | 1e-3 | 0.7 |
| Clothing | 5e-4 | 2 | 0.50 | 0.45 | 1.10 | 0.0 | 0.7 |
| MicroLens | 1e-3 | 1 | 0.03 | 0.25 | 1.10 | 0.0 | 0.0 |

Sports and Clothing retain 30% of aggregated user-item edges per epoch. Edge
selection follows the existing degree-sensitive multinomial policy using the
complete normalized temporal graph as sampling scores. The retained edges keep
their original raw temporal weights and are symmetrically renormalized before
message passing. Evaluation always uses the complete normalized graph.
