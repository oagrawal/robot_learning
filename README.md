# RL

Quick start for setting up this repo on a new machine.

## Prerequisites

- Python 3.10+
- pip
- Git
- CUDA-capable GPU (e.g. A100) recommended for training

## Setup

### 1. Create and activate a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
```

### 2. Download the HDF5 dataset

```bash
pip install gdown  # if not already installed
gdown 1paIqhdqswq4-a-KZoBGey0BH9s7x_k9B
```

This downloads `square_d0.hdf5` (~1.6 GB) to the current directory.

### 3. Install dependencies (monorepo)

This repo already contains `mimicgen/`, `robomimic/`, and `imitation/` as subdirectories.
From the repo root (where `square_d0.hdf5` lives), just install the packages:

```bash
# robosuite from PyPI
pip install robosuite

# local packages from this repo (editable mode)
pip install -e mimicgen
pip install -e robomimic
pip install -e imitation
```

### 4. Edit the config before training

**Important:** Update `imitation/imitation/config/model/flow_policy_config.py`:

1. **`output_dir`** in `train_config` – set to your experiments path, e.g.:
   ```python
   output_dir="/path/to/your/repo/experiments"
   ```

2. **`data`** in `data_config` – set to your HDF5 path, e.g.:
   ```python
   data=["/path/to/your/repo/square_d0.hdf5"]
   ```

Replace `/path/to/your/repo` with the absolute path to this repo on your machine.

### 5. (Optional) wandb login

For training logs:

```bash
wandb login
```

Paste your 40-character API key from https://wandb.ai/authorize (the key only, not the URL).

## Running

**Option A: Jupyter notebook**

1. Update `DATASET_PATH` and `BASE_DIR` in the first cell of `Imitation-Learning-with-Negative-Examples.ipynb` to match your paths.
2. Run the notebook. Skip cells marked "SKIP" if mimicgen/robomimic/imitation are already cloned and installed.
3. Skip the "Blackwell PyTorch" cell if you have A100 or other standard GPUs.

**Option B: Command line**

```bash
cd imitation/imitation
python scripts/train.py --exp_name baseline_square --config config/model/flow_policy_config.py
```

## Notes

- **GPU:** A100 and similar GPUs use standard PyTorch; no need for the Blackwell nightly build.
- **num_workers:** In `flow_policy_config.py`, `num_workers` in `data_config` can be tuned to your CPU core count (default 16).
