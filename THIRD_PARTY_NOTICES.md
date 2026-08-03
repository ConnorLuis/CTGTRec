# Third-Party Notices and Provenance

This file records the principal upstream software and data provenance relevant
to the CTGTRec repository. It is not a replacement for the license text in
`LICENSE` or for notices retained inside individual source files.

## MMRec

- Project: **MMRec — a multimodal recommendation toolbox**
- Upstream repository: <https://github.com/enoche/MMRec>
- Upstream license: **GNU General Public License version 3**
- Relationship to CTGTRec: CTGTRec uses and modifies the MMRec training,
  configuration, data-loading, evaluation, utility, and baseline-model
  framework.

The combined modified source is released under `GPL-3.0-only`. Original MMRec
copyright, author, and email notices that remain in inherited files should not
be removed.

The following are CTGTRec-side modifications or additions made in 2026:

- continuous-time weighted user-item graph construction;
- item temporal trend estimation and additive score calibration;
- strict train/validation/test temporal preprocessing;
- weighted graph dropout that preserves temporal edge weights;
- validation-only checkpoint selection and one-time final test evaluation;
- fixed seeds `999`, `2024`, and `3407`, with mean and sample-standard-deviation
  aggregation;
- project-specific configuration, command-line, documentation, and release
  cleanup;
- additional runnable baseline adapters.

## Individual Recommendation Methods

The repository contains implementations or adapters for recommendation methods
described in their respective papers, including:

```text
BPR-MF, LightGCN,
VBPR, MMGCN, GRCN, LATTICE, BM3, SLMRec, MGCN, FREEDOM,
MISSRec, HM4SR, M3Rec, MuSTRec,
TimeMM
```

Some implementations are inherited from MMRec; some are repository-specific
adapters. Consult:

- comments and attribution headers in each file under `src/models/`;
- the corresponding YAML under `src/configs/model/`;
- the cited paper;
- the original implementation repository when one is identified.

Do not remove upstream notices from files that contain them. When
redistributing a model implementation copied or adapted from another source,
also comply with that source's license and attribution requirements.

## Datasets and Feature Artifacts

CTGTRec experiments use:

- Amazon Review Data subsets: Baby, Sports, and Clothing;
- MicroLens short-video recommendation data;
- pre-extracted visual and textual feature artifacts supplied by or derived
  from the corresponding data sources.

The repository does not grant rights to the underlying datasets, images, text,
videos, or pretrained feature artifacts. Users must obtain them from their
original providers and comply with their licenses, terms of use, privacy
requirements, and citation instructions.

Dataset instructions are documented in `data/README.md` and
`preprocessing/raw/README.md`.

## Python Dependencies

Packages installed from `requirements.txt` and
`requirements-preprocessing.txt` retain their own licenses. Installing or
using those packages does not relicense them under GPLv3. Review the package
metadata and upstream repositories when redistributing an environment or
binary bundle.

## No Warranty

The software is provided without warranty under the terms of GPLv3. This notice
is informational and is not legal advice.
