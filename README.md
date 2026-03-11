# robot_learning

Evaluation of a Flow Matching policy trained on the `square_d0` MimicGen dataset using the [`imitation`](https://github.com/ShivinDass/imitation) repo.

## Environment Setup

```bash
# Create a Python 3.10 venv (required — pinned deps don't have py3.12 wheels)
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate
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

# Install imitation as editable
pip install -e imitation/
```

## Running Evaluation

```bash
python eval_square_d0.py \
    --checkpoint weights_ep900.pth \
    --n_rollouts 20 \
    --max_steps 400 \
    --save_video \
    --video_dir rollout_videos
```

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
