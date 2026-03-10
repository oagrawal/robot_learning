# RL

Quick start for setting up this repo on a new machine.

## Prerequisites

- Python 3.10+
- pip
- Git
- CUDA-capable GPU (e.g. A100) recommended for training

## Setup

### 1. Create and activate a Python 3.10 virtual environment

**Required:** Use Python 3.10 (Python 3.14+ is not supported—mujoco and other deps lack wheels).

```bash
# If python3.10 is not installed: brew install python@3.10
python3.10 -m venv venv
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
# robosuite (robot simulation framework) - clone for editable install so you can modify if needed
cd robosuite && pip install -e . && cd ..

# local packages from this repo (editable mode)
pip install -r imitation/requirements.txt
pip install -e mimicgen
pip install -e robomimic
pip install -e imitation
```


### 5. (Optional) wandb login

For training logs:

```bash
pip install --upgrade wandb
wandb login
```

Paste your 40-character API key from https://wandb.ai/authorize (the key only, not the URL).


