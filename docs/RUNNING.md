# Running CTGTRec

All public commands are intended to be executed from the repository root.

## 1. Create the environment

Reference interpreter:

```text
Python 3.10
```

Example virtual environment:

```bash
python -m venv .venv
```

Activate it, then install the CUDA 12.8 PyTorch build:

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

## 2. Prepare the data and graph files

The expected dataset directory is:

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

See `preprocessing/README.md` for strict temporal splitting and continuous-time
graph construction.

## 3. Check the merged configuration

From the repository root:

```bash
python src/main.py \
  --model CTGTRec \
  --dataset baby \
  --show-config
```

The printed paths must point inside the current repository, including:

```text
project_root=<repository>
data_path=<repository>/data/
checkpoint_dir=<repository>/saved
result_dir=<repository>/results
log_dir=<repository>/logs
```

Configuration files are discovered from `src/configs/` based on the location of
`configurator.py`, not the current shell directory.

## 4. Run the final three-seed experiment

Baby:

```bash
python src/main.py --model CTGTRec --dataset baby
```

Sports:

```bash
python src/main.py --model CTGTRec --dataset sports
```

Clothing:

```bash
python src/main.py --model CTGTRec --dataset clothing
```

MicroLens:

```bash
python src/main.py --model CTGTRec --dataset microlens
```

Each command runs seeds `999`, `2024`, and `3407`, restores each seed's best
validation checkpoint, evaluates the test split once per seed, and writes
per-seed plus aggregate outputs.

## 5. Command-line controls

Select a GPU:

```bash
python src/main.py --model CTGTRec --dataset baby --gpu-id 1
```

Force CPU execution:

```bash
python src/main.py --model CTGTRec --dataset baby --cpu
```

Disable checkpoint files while retaining evaluation outputs:

```bash
python src/main.py --model CTGTRec --dataset baby --no-save-model
```

Use temporary debugging overrides:

```bash
python src/main.py \
  --model CTGTRec \
  --dataset baby \
  --set epochs=2 \
  --set stopping_step=1
```

`--set` is intended for debugging. Do not use it when reproducing the fixed
paper configurations.

## 6. Output locations

All paths are rooted at the repository regardless of the shell's initial
working directory:

```text
logs/
saved/
results/
recommend_topk/
```

The standard CTGTRec aggregate result is written to:

```text
results/ctgtrec/<dataset>/combo_000/
├── seed_results.csv
├── summary.csv
└── summary.json
```

## 7. Failure behavior

The entry point exits with an explicit error when:

- a required overall, dataset, or model YAML is missing;
- `--set` does not use `KEY=VALUE`;
- an invalid GPU/thread value is supplied;
- required project path settings are absent;
- a configuration key reserved by the entry point is overridden.

Unknown command-line options are rejected rather than silently ignored.
