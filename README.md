# robot_learning

Evaluation of a Flow Matching policy trained on the `square_d0` MimicGen dataset using the [`imitation`](https://github.com/ShivinDass/imitation) repo.

## Environment Setup

```bash
# Create a Conda environment with Python 3.10 (required — pinned deps don't have py3.12 wheels)
conda create -n flow-learning python=3.10 -y
conda activate flow-learning
```

## Git LFS (Large File Storage)

This repository uses **Git LFS** to store model weights (`.pth` files). Standard Git is not designed for large binary files; LFS replaces these files with tiny text pointers in GitHub and downloads the actual data only when needed.

### Setup for New Users

Before you can see the actual weight files, you must install and initialize Git LFS on your machine:

**On Linux (Ubuntu):**
```bash
sudo apt install git-lfs
git lfs install
```

**On macOS:**
```bash
brew install git-lfs
git lfs install
```

**After installation**, download the actual weights into your local directory:
```bash
git lfs pull
```


## Install Dependencies

```bash
# Core imitation dependencies (pinned versions)
pip install -r imitation/requirements.txt
# torch==2.7.0, numpy==1.22.4, torchvision==0.22.0, diffusers==0.33.1,
# wandb==0.13.1, h5py, einops

# Robosuite v1.4 (must be this version — imitation/ was written for v1.4 API)
pip install "robosuite==1.4.0"

# Gymnasium v0.29.1 (must be this version — v1.x has breaking Wrapper assert)
pip install "gymnasium==0.29.1"

# Media / rendering
pip install imageio imageio-ffmpeg

# Robosuite undeclared dependency
pip install termcolor

# Address setuptools and numpy dependency breaks
pip install "setuptools<70.0.0"
pip install "numpy<2" "opencv-python<4.10"

# Install imitation as editable
pip install -e imitation/
```

## Download the Dataset / Checkpoints

```bash
pip install gdown
gdown 1paIqhdqswq4-a-KZoBGey0BH9s7x_k9B  # Download square_d0.hdf5 dataset

# Move the dataset logically inside the imitation module
mkdir -p imitation/imitation/data
mv square_d0.hdf5 imitation/imitation/data/
```

## Running Evaluation

### Standard Evaluation
```bash
python eval_square_d0.py \
    --checkpoint weights_ep900.pth \
    --n_rollouts 20 \
    --max_steps 400 \
    --save_video \
    --video_dir rollout_videos
```

### Minimal Evaluation (Hardcoded Config)
```bash
python minimal_eval.py
```

## Running Training

To configure your Python path and train the Flow Matching model on the `square_d0` dataset:

```bash
# Execute the training script
python imitation/imitation/scripts/train.py \
    --exp_name baseline_square \
    --config imitation/imitation/config/model/flow_policy_config.py
```

### Running Remotely in Background (`tmux`)

To ensure training survives SSH disconnections:
1. Start a new session: `tmux new -s flow_training`
2. Run the training command above
3. Detach and leave it running: Press `Ctrl+B`, release, then press `D`
4. Reattach later: `tmux attach -t flow_training`

## Key Dependency Notes

| Package | Version | Reason |
|---|---|---|
| Python | 3.10 | Required — pinned deps lack py3.12 wheels |
| torch | 2.7.0 | Pinned in `imitation/requirements.txt` |
| numpy | 1.22.4 | Pinned in `imitation/requirements.txt` |
| diffusers | 0.33.1 | Later versions removed re-exports used by `lr_scheduler.py` |
| robosuite | 1.4.0 | `imitation/` uses v1.4 API (`suite.make`, `load_controller_config`) |
| gymnasium | 0.29.1 | v1.x added a strict `isinstance` check that breaks `RobosuiteImageFlipWrapper` |
| termcolor | any | Used by robosuite but not declared in its dependencies |
