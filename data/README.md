# Datasets

CTGTRec is evaluated on four temporal multimodal recommendation datasets:

- **Baby**
- **Sports**
- **Clothing**
- **MicroLens**

The Baby, Sports, and Clothing datasets are Amazon product review subsets.
MicroLens is a short-video recommendation dataset.

## Download

### Amazon Datasets

Download the Baby, Sports, and Clothing datasets from:

[Google Drive: Baby / Sports / Clothing](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)

### MicroLens

Download the MicroLens dataset from:

[Google Drive: MicroLens](https://drive.google.com/drive/folders/14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w?usp=drive_link)

We thank [@yxni98](https://github.com/yxni98) for providing the processed
MicroLens data.

## Multimodal Features

The downloaded datasets include pre-extracted visual and textual item
features.

| Dataset | Visual feature dimension | Textual feature dimension |
| --- | ---: | ---: |
| Baby | 4096 | 384 |
| Sports | 4096 | 384 |
| Clothing | 4096 | 384 |
| MicroLens | 1024 | 1024 |

## Data Placement

Place the downloaded datasets under this `data/` directory before running
the preprocessing and training scripts.

The exact directory structure and preprocessing commands will be described
in the preprocessing documentation.

## Data Split

The repository applies a per-user chronological split during preprocessing:

- the last interaction is used for testing;
- the second-to-last interaction is used for validation;
- all earlier interactions are used for training.

When multiple interactions from the same user have identical timestamps,
their original record order is used as the secondary sorting key.

Only training interactions are used to construct graph structures and item
trend statistics.

## License

This repository does not redistribute the original datasets. Please download
them from the links above and comply with the corresponding dataset licenses
and terms of use.