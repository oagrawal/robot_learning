"""
Live visualizer for the trained trajectory classifier on the TEST split.

For each test-set demo (positives + negatives derived from the same
three_way_hard_val split used at train time) this plays the agentview video
frame-by-frame and overlays the classifier's output at every timestep:
    - P(success) in [0, 1]
    - hard prediction: 1 (success) if P >= threshold else 0 (failure)

Expected behavior:
    - success demos -> predictions should be 1 for all windows.
    - failure demos -> predictions should be 1s pre-divergence, 0s after the
      critical point (i.e. the classifier should "flip" around the transition).

Usage:
    cd imitation/imitation
    python data/visualize_classifier_predictions.py \\
        --config ./config/model/classifier_config.py \\
        --checkpoint /abs/path/to/best_classifier.pth

Optional:
    --threshold 0.5           decision threshold for the 0/1 label
    --fps 6.0                 autoplay fps
    --start_demo 0            start index within the combined test demo list
    --only {success,failure}  restrict playback to one class
    --device cuda|cpu

Keybinds (window must be focused):
    , or k        Prev frame
    . or l        Next frame
    Space         Toggle autoplay
    n / p         Next / prev test demo
    q or ESC      Quit
"""

from __future__ import annotations

import argparse
import os
from importlib.machinery import SourceFileLoader
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch

from imitation.models.trajectory_classifier import TrajectoryClassifier
from imitation.utils.file_utils import (
    get_all_obs_keys_from_config,
    get_obs_key_to_modality_from_config,
)
from imitation.utils.obs_utils import process_obs_dict
from imitation.utils.tensor_utils import recursive_dict_list_tuple_apply


def _build_test_dataset(conf):
    """Instantiate the ClassifierDataset with split='test' using the same
    split_strategy the model was trained with."""
    data_config = conf.data_config
    obs_key_to_modality = get_obs_key_to_modality_from_config(conf.observation_config)

    ds_kwargs = dict(data_config.dataset_kwargs)
    ds_kwargs['obs_keys_to_modality'] = obs_key_to_modality
    ds_kwargs['obs_keys_to_normalize'] = conf.observation_config.obs_keys_to_normalize

    test_dataset = data_config.dataset_class(
        data_paths=data_config.data, split='test', **ds_kwargs
    )
    return test_dataset, obs_key_to_modality


def _iter_test_demos(dataset, only: Optional[str]) -> List[Tuple[int, str, float]]:
    """Return ordered (file_idx, demo_key, label) for the test split.

    Preserves the 'success first, then failure' ordering used in
    _build_index_three_way_hard_val so the visualizer is deterministic.
    """
    sd = getattr(dataset, 'split_demos', None)
    if sd is None:
        # Fall back to index introspection for older split strategies.
        seen: Dict[Tuple[int, str], float] = {}
        for fi, dk, _t, label in dataset.index:
            seen.setdefault((fi, dk), float(label))
        return [(fi, dk, lab) for (fi, dk), lab in seen.items()]

    pos = [(fi, dk, 1.0) for fi, dk, _ in sd['pos_test']]
    neg = [(fi, dk, 0.0) for fi, dk, _ in sd['neg_test']]
    if only == 'success':
        return pos
    if only == 'failure':
        return neg
    return pos + neg


def _load_agentview(dataset, file_idx: int, demo_key: str) -> np.ndarray:
    """Load the agentview_image stream for a demo from the open HDF5 file."""
    path = dataset.hdf5_paths[file_idx]
    with h5py.File(path, 'r', swmr=True, libver='latest') as f:
        demo = f[f'data/{demo_key}']
        if 'obs/agentview_image' in demo:
            return demo['obs/agentview_image'][:]
        if 'obs' in demo and 'agentview_image' in demo['obs']:
            return demo['obs']['agentview_image'][:]
        raise KeyError(
            f"Demo {demo_key} in {path} has no obs/agentview_image for visualization"
        )


def _compute_demo_probs(model, dataset, obs_key_to_modality, file_idx, demo_key, device, batch_size=64):
    """Return a (T,) array of P(success) for every timestep of the demo."""
    traj = dataset.get_trajectory_data(file_idx, demo_key)
    n_windows = traj['actions'].shape[0]
    probs = np.zeros(n_windows, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            obs_batch = {k: traj['obs'][k][start:end] for k in traj['obs']}
            act_batch = traj['actions'][start:end]

            obs_batch = process_obs_dict(obs_batch, obs_key_to_modality)
            obs_batch = recursive_dict_list_tuple_apply(
                obs_batch, {np.ndarray: lambda x: torch.from_numpy(x).float().to(device)}
            )
            act_t = torch.from_numpy(act_batch).float().to(device)

            p = model.predict_prob(obs_batch, act_t).squeeze(-1).cpu().numpy()
            probs[start:end] = p
    return probs


def _render_trace(probs: np.ndarray, t: int, threshold: float, width: int, height: int) -> np.ndarray:
    """Render a P(success) trace of shape (height, width, 3) with a cursor at t."""
    img = np.full((height, width, 3), 30, dtype=np.uint8)

    if len(probs) <= 1:
        xs = np.zeros(len(probs), dtype=np.int32)
    else:
        xs = np.linspace(0, width - 1, len(probs)).astype(np.int32)
    ys = (height - 1 - probs * (height - 1)).astype(np.int32)

    # 0.5 threshold line
    thr_y = int(round(height - 1 - threshold * (height - 1)))
    cv2.line(img, (0, thr_y), (width - 1, thr_y), (80, 80, 80), 1, cv2.LINE_AA)

    # Color trace: green where P>=threshold (success), red otherwise.
    for i in range(1, len(probs)):
        color = (0, 200, 0) if probs[i] >= threshold else (0, 0, 220)
        cv2.line(img, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), color, 2, cv2.LINE_AA)

    # Current-timestep cursor.
    if len(probs) > 0:
        cx = int(xs[min(t, len(probs) - 1)])
        cv2.line(img, (cx, 0), (cx, height - 1), (255, 255, 0), 1, cv2.LINE_AA)
        cy = int(ys[min(t, len(probs) - 1)])
        cv2.circle(img, (cx, cy), 4, (255, 255, 0), -1, cv2.LINE_AA)

    cv2.putText(img, '1.0', (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, '0.0', (2, height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(
        img, f'thr={threshold:.2f}',
        (width - 80, thr_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA,
    )
    return img


def _compose_frame(
    rgb: np.ndarray,
    probs: np.ndarray,
    t: int,
    *,
    label: float,
    demo_key: str,
    file_idx: int,
    demo_index: int,
    n_demos: int,
    threshold: float,
    autoplay: bool,
    fps: float,
) -> np.ndarray:
    frame = rgb.copy()
    h, w = frame.shape[:2]
    scale = max(1, 480 // max(h, 1))
    frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    h, w = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    total = len(probs)
    t_clamped = max(0, min(t, total - 1)) if total > 0 else 0
    p = float(probs[t_clamped]) if total > 0 else 0.0
    pred = 1 if p >= threshold else 0
    gt_tag = 'SUCCESS' if label > 0.5 else 'FAILURE'
    pred_tag = 'SUCCESS' if pred == 1 else 'FAILURE'
    correct = (pred == 1 and label > 0.5) or (pred == 0 and label <= 0.5)

    # Border color: green if classifier matches ground truth, red otherwise.
    border_color = (0, 220, 0) if correct else (0, 0, 220)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 3)

    lines = [
        f"{demo_key}  ({demo_index + 1}/{n_demos})  file_idx={file_idx}  [{gt_tag}]",
        f"t = {t_clamped} / {max(total - 1, 0)}   (len={total})",
        f"P(success) = {p:.3f}   pred={pred} ({pred_tag})   thr={threshold:.2f}",
        f"autoplay: {'ON' if autoplay else 'OFF (Space=play, . or l=step)'}  ({fps:.1f} fps)",
        ", . k l step   Space play   n/p demo   q quit",
    ]
    y = 18
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    trace = _render_trace(probs, t_clamped, threshold, width=w, height=120)
    return np.vstack([frame, trace])


def run(args: argparse.Namespace) -> None:
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Loading classifier config: {args.config}")
    conf = SourceFileLoader('conf', args.config).load_module().config

    # Sanity check: visualizer only supports the leakage-free three_way split.
    strategy = conf.data_config.dataset_kwargs.get('split_strategy')
    if strategy != 'three_way_hard_val':
        print(
            f"WARNING: split_strategy={strategy!r}; the visualizer was designed for "
            f"'three_way_hard_val'. Proceeding anyway using whatever split='test' returns."
        )

    print("Building test dataset (this reproduces the exact train/val/test split)...")
    test_dataset, obs_key_to_modality = _build_test_dataset(conf)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = TrajectoryClassifier.load(os.path.expanduser(args.checkpoint), device=device)
    model = model.to(device)
    model.eval()

    demos = _iter_test_demos(test_dataset, args.only)
    if not demos:
        print(f"No test demos found (only={args.only}).")
        return
    print(f"Test demos to visualize: {len(demos)}")

    demo_idx = max(0, min(int(args.start_demo), len(demos) - 1))
    t = 0
    autoplay = bool(args.autoplay)
    fps = max(0.5, float(args.fps))
    delay_ms = max(1, int(1000.0 / fps))
    threshold = float(args.threshold)

    win = 'Classifier live predictions (TEST split)'

    # Per-demo caches so we don't re-run the model on every key press.
    image_cache: Dict[Tuple[int, str], np.ndarray] = {}
    prob_cache: Dict[Tuple[int, str], np.ndarray] = {}

    def load_demo(fi: int, dk: str):
        key = (fi, dk)
        if key not in image_cache:
            image_cache[key] = _load_agentview(test_dataset, fi, dk)
        if key not in prob_cache:
            print(f"  scoring {dk} (file_idx={fi})...")
            prob_cache[key] = _compute_demo_probs(
                model, test_dataset, obs_key_to_modality, fi, dk, device
            )
        return image_cache[key], prob_cache[key]

    while True:
        file_idx, demo_key, label = demos[demo_idx]
        images, probs = load_demo(file_idx, demo_key)
        total = len(images)
        t = max(0, min(t, total - 1))

        frame = _compose_frame(
            images[t], probs, t,
            label=label,
            demo_key=demo_key,
            file_idx=file_idx,
            demo_index=demo_idx,
            n_demos=len(demos),
            threshold=threshold,
            autoplay=autoplay,
            fps=fps,
        )
        cv2.imshow(win, frame)

        wait = delay_ms if autoplay else 0
        raw = cv2.waitKey(wait)
        key = 0 if raw == -1 else (raw & 0xFF)

        if autoplay and key == 0:
            t = (t + 1) % total

        if key == 27 or key == ord('q'):
            print("Quit.")
            break

        if key == ord(' ') or key == 32:
            autoplay = not autoplay
            print(f"Autoplay: {autoplay}")
        elif key == ord(',') or key == ord('k'):
            t = max(0, t - 1)
        elif key == ord('.') or key == ord('l'):
            t = min(total - 1, t + 1)
        elif key == ord('n'):
            demo_idx = min(len(demos) - 1, demo_idx + 1)
            t = 0
            print(f"Demo -> ({demo_idx + 1}/{len(demos)}) {demos[demo_idx][1]}")
        elif key == ord('p'):
            demo_idx = max(0, demo_idx - 1)
            t = 0
            print(f"Demo -> ({demo_idx + 1}/{len(demos)}) {demos[demo_idx][1]}")

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Live classifier predictions over TEST-split demos'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to classifier_config.py used for training')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to classifier .pth checkpoint (best or final)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for the hard 0/1 label')
    parser.add_argument('--fps', type=float, default=6.0, help='Autoplay FPS')
    parser.add_argument('--start_demo', type=int, default=0,
                        help='Index into the test-demo list to start at')
    parser.add_argument('--only', type=str, default=None,
                        choices=['success', 'failure'],
                        help='Restrict playback to one class (default: both)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--autoplay', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
