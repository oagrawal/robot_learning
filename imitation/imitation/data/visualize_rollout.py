"""
Visualize HDF5 rollouts with overlaid reward / label information.

Plays the agentview_image stream for a given demo and overlays:
  - Staged rewards (reach / grasp / lift / hover) as coloured bars
  - Per-timestep c_labels if present (green = success, red = failure)
  - Current timestep counter

Usage:
    # Play demo 0 from the failure dataset (display in a window)
    python visualize_rollout.py --input ./data/failure_labeled.hdf5 --demo 0

    # Save demo 5 as an MP4 without displaying
    python visualize_rollout.py --input ./data/failure_labeled.hdf5 --demo 5 --save rollout_5.mp4

    # Play all demos one-by-one (press Q to skip, ESC to quit)
    python visualize_rollout.py --input ./data/failure_labeled.hdf5 --all

    # Play with 2x speed
    python visualize_rollout.py --input ./data/failure_labeled.hdf5 --demo 0 --speed 2
"""

import os
import argparse
import numpy as np
import h5py
import cv2


def draw_overlay(frame, step, total_steps, staged_rewards=None, c_label=None):
    """Draw informational overlay on a frame."""
    # Scale up for readability (84x84 is tiny)
    h, w = frame.shape[:2]
    scale = max(1, 480 // h)
    frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    h, w = frame.shape[:2]

    # Convert RGB → BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --- Timestep counter ---
    cv2.putText(frame, f"t={step}/{total_steps}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- c_label indicator ---
    if c_label is not None:
        if c_label > 0.5:
            label_text = "c=1 (success)"
            color = (0, 200, 0)  # green
        else:
            label_text = "c=0 (failure)"
            color = (0, 0, 200)  # red
        cv2.putText(frame, label_text, (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # --- Staged reward bars ---
    if staged_rewards is not None:
        bar_names = ["reach", "grasp", "lift", "hover"]
        bar_colors = [
            (200, 200, 50),   # reach - cyan-ish
            (50, 200, 50),    # grasp - green
            (50, 150, 200),   # lift - orange-ish
            (200, 50, 200),   # hover - magenta
        ]
        bar_x = 8
        bar_y_start = h - 80
        bar_width = 50
        bar_max_height = 60

        for i, (name, color) in enumerate(zip(bar_names, bar_colors)):
            val = float(staged_rewards[i])
            bx = bar_x + i * (bar_width + 8)
            bar_height = int(val * bar_max_height)

            # Background
            cv2.rectangle(frame, (bx, bar_y_start), (bx + bar_width, bar_y_start + bar_max_height),
                          (60, 60, 60), -1)
            # Filled bar
            if bar_height > 0:
                cv2.rectangle(frame, (bx, bar_y_start + bar_max_height - bar_height),
                              (bx + bar_width, bar_y_start + bar_max_height),
                              color, -1)
            # Label
            cv2.putText(frame, f"{name}", (bx, bar_y_start + bar_max_height + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{val:.2f}", (bx, bar_y_start - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return frame


def play_demo(f, demo_key, speed=1, save_path=None):
    """Play or save a single demo from the HDF5 file."""
    demo = f[f'data/{demo_key}']

    # Get frames
    if 'obs/agentview_image' in demo:
        images = demo['obs/agentview_image'][:]
    elif 'obs' in demo and 'agentview_image' in demo['obs']:
        images = demo['obs']['agentview_image'][:]
    else:
        print(f"  No agentview_image found in {demo_key}. Available keys:")
        def print_keys(group, prefix=""):
            for k in group.keys():
                if isinstance(group[k], h5py.Group):
                    print(f"    {prefix}{k}/")
                    print_keys(group[k], prefix + "  ")
                else:
                    print(f"    {prefix}{k}: {group[k].shape}")
        print_keys(demo)
        return False

    total_steps = len(images)

    # Get optional data
    staged_rewards = None
    if 'staged_rewards' in demo:
        staged_rewards = demo['staged_rewards'][:]

    c_labels = None
    if 'c_labels' in demo:
        c_labels = demo['c_labels'][:]

    print(f"\n  Playing {demo_key}: {total_steps} steps", end="")
    if staged_rewards is not None:
        peak_hover = np.max(staged_rewards[:, 3])
        print(f" | peak hover: {peak_hover:.3f}", end="")
    if c_labels is not None:
        n_pos = np.sum(c_labels > 0.5)
        n_neg = np.sum(c_labels <= 0.5)
        print(f" | labels: {int(n_pos)} pos / {int(n_neg)} neg", end="")
    print()

    # Setup video writer if saving
    writer = None
    if save_path:
        # Render first frame to get dimensions
        sample = draw_overlay(images[0], 0, total_steps,
                              staged_rewards[0] if staged_rewards is not None else None,
                              c_labels[0] if c_labels is not None else None)
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, 30, (w, h))

    fps = 30 * speed
    delay = max(1, int(1000 / fps))

    for step in range(total_steps):
        sr = staged_rewards[step] if staged_rewards is not None else None
        cl = c_labels[step] if c_labels is not None else None

        frame = draw_overlay(images[step], step, total_steps, sr, cl)

        if writer:
            writer.write(frame)
        else:
            cv2.imshow(f"Rollout: {demo_key}", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                print("  Skipping...")
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return True  # signal quit

    if writer:
        writer.release()
        print(f"  Saved to {save_path}")
    else:
        # Show last frame until key press
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return False  # don't quit


def main():
    parser = argparse.ArgumentParser(description="Visualize HDF5 rollouts with reward/label overlays")
    parser.add_argument('--input', type=str, required=True, help='Path to HDF5 file')
    parser.add_argument('--demo', type=int, default=None, help='Demo index to play (e.g. 0 for demo_0)')
    parser.add_argument('--all', action='store_true', help='Play all demos sequentially')
    parser.add_argument('--save', type=str, default=None, help='Save as MP4 instead of displaying')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (default: 1)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save all demo videos (used with --all)')
    args = parser.parse_args()

    with h5py.File(args.input, 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
        print(f"Found {len(demos)} demos in {args.input}")

        # Print HDF5 structure for the first demo
        first_demo = f[f'data/{demos[0]}']
        print(f"\nStructure of {demos[0]}:")
        for k in first_demo.keys():
            if isinstance(first_demo[k], h5py.Group):
                print(f"  {k}/")
                for kk in first_demo[k].keys():
                    print(f"    {kk}: {first_demo[k][kk].shape}")
            else:
                print(f"  {k}: {first_demo[k].shape}")

        if args.all:
            for demo_key in demos:
                save_path = None
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    save_path = os.path.join(args.save_dir, f"{demo_key}.mp4")
                quit_signal = play_demo(f, demo_key, speed=args.speed, save_path=save_path)
                if quit_signal:
                    break
        elif args.demo is not None:
            demo_key = f"demo_{args.demo}"
            if demo_key not in demos:
                print(f"Error: {demo_key} not found. Available: {demos[:5]}...")
                return
            play_demo(f, demo_key, speed=args.speed, save_path=args.save)
        else:
            print("\nSpecify --demo N or --all to play demos.")
            print(f"Example: python visualize_rollout.py --input {args.input} --demo 0")


if __name__ == '__main__':
    main()
