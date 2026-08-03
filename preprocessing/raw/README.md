# Raw-data preprocessing

These command-line scripts replace the four legacy preprocessing notebooks:

| Legacy notebook | Public script |
| --- | --- |
| `0rating2inter.ipynb` | `build_interactions.py` |
| `1splitting.ipynb` | `split_interactions.py` |
| `2reindex-feat.ipynb` | `reindex_amazon_metadata.py` |
| `3feat-encoder.ipynb` | `encode_amazon_features.py` |

Run all commands from the CTGTRec repository root.

## Scope

The metadata and feature scripts reproduce the Amazon preprocessing represented
by the notebooks. They do not reconstruct MicroLens visual/textual encoders from
raw video files. For MicroLens, obtain the official interactions and extracted
modality features from the dataset maintainers, then apply the CTGTRec temporal
split to the interaction file.

The scripts never download or redistribute dataset files. Users must follow the
terms and citation requirements of the original providers:

- Amazon Review Data: <https://nijianmo.github.io/amazon/index.html>
- MicroLens: <https://github.com/westlake-repl/MicroLens>

## Dependencies

After installing the main environment, install the optional raw text-encoding
dependency set:

```bash
pip install -r requirements-preprocessing.txt
```

The interaction, split, metadata, and image-alignment stages themselves use
NumPy and pandas from `requirements.txt`. Text encoding uses the pinned
Sentence Transformers package from `requirements-preprocessing.txt`.

## 1. Build interactions and mappings

### Amazon ratings-only CSV

The Amazon pages have used more than one documented column order. The original
2014 notebook read:

```text
user, item, rating, timestamp
```

The newer Amazon Review Data page describes ratings-only tuples as:

```text
item, user, rating, timestamp
```

The script therefore requires the order to be stated explicitly rather than
silently guessing it.

Legacy notebook-compatible example:

```bash
python preprocessing/raw/build_interactions.py \
  --source_type amazon-ratings-csv \
  --input raw/ratings_Sports_and_Outdoors.csv \
  --amazon_column_order user-item-rating-timestamp \
  --output_dir data/sports \
  --dataset sports
```

Newer documented order:

```bash
python preprocessing/raw/build_interactions.py \
  --source_type amazon-ratings-csv \
  --input raw/Sports_and_Outdoors.csv \
  --amazon_column_order item-user-rating-timestamp \
  --output_dir data/sports \
  --dataset sports
```

### Amazon review JSON-lines

```bash
python preprocessing/raw/build_interactions.py \
  --source_type amazon-reviews-jsonl \
  --input raw/Sports_and_Outdoors.json.gz \
  --output_dir data/sports \
  --dataset sports
```

### MicroLens interaction CSV/TSV

Specify the actual column names from the downloaded release:

```bash
python preprocessing/raw/build_interactions.py \
  --source_type microlens-csv \
  --input raw/MicroLens_interactions.csv \
  --delimiter , \
  --user_column userID \
  --item_column videoID \
  --timestamp_column timestamp \
  --output_dir data/microlens \
  --dataset microlens
```

If the MicroLens release contains an explicit feedback/rating column, add
`--rating_column <name>`. Otherwise the script assigns implicit rating `1.0`.

The default filtering is iterative 5-core for users and items. Outputs are:

```text
data/<dataset>/
├── <dataset>.inter
├── u_id_mapping.csv
├── i_id_mapping.csv
└── raw_preprocessing_manifest.json
```

The `.inter` file initially has `x_label = 0` for every row.

## 2. Apply the strict CTGTRec temporal split

Recommended multi-dataset command:

```bash
python preprocessing/build_temporal_split_inter.py \
  --data_root data \
  --datasets baby sports clothing microlens
```

The notebook-compatible single-file entry point is also available:

```bash
python preprocessing/raw/split_interactions.py \
  --input data/sports/sports.inter \
  --output data/sports/sports_temporal.inter
```

Both commands use the same canonical implementation. For each user,
interactions are ordered by `(timestamp, original_record_order)`; the last is
test, the second-to-last is validation, and all earlier rows are training.

## 3. Align Amazon metadata to item IDs

```bash
python preprocessing/raw/reindex_amazon_metadata.py \
  --item_mapping data/sports/i_id_mapping.csv \
  --metadata raw/meta_Sports_and_Outdoors.json.gz \
  --output data/sports/meta-sports.csv \
  --missing_output data/sports/missing_metadata_itemIDs.csv
```

The default `--missing_policy error` prevents silent row misalignment. Use
`--missing_policy empty` only when you deliberately want empty rows for items
with no metadata.

The parser accepts both modern JSON-lines files and older Python dictionary
literal files. It never uses `eval`.

## 4. Encode Amazon features

### Text features

```bash
python preprocessing/raw/encode_amazon_features.py \
  --mode text \
  --metadata data/sports/meta-sports.csv \
  --output_dir data/sports \
  --device cuda:0
```

The default text protocol reproduces the notebook:

```text
title + brand + first category path + description
```

with `all-MiniLM-L6-v2`, producing `text_feat.npy` with 384 columns.

### Image features

```bash
python preprocessing/raw/encode_amazon_features.py \
  --mode image \
  --metadata data/sports/meta-sports.csv \
  --image_binary raw/image_features_Sports_and_Outdoors.b \
  --output_dir data/sports
```

The expected binary format is one 10-byte ASIN followed by 4096 float32 values.
Items missing from the binary are filled with the mean available mapped image
vector, matching the notebook. The command writes:

```text
data/sports/image_feat.npy
data/sports/missed_img_itemIDs.csv
```

### Text and image together

```bash
python preprocessing/raw/encode_amazon_features.py \
  --mode all \
  --metadata data/sports/meta-sports.csv \
  --image_binary raw/image_features_Sports_and_Outdoors.b \
  --output_dir data/sports \
  --device cuda:0
```

## Re-running commands

All scripts refuse to overwrite existing outputs by default. Add `--overwrite`
only after checking the destination paths.
