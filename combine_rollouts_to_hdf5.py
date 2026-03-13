"""
Combines rollout .npz files produced by minimal_eval.py into two HDF5 files
in robomimic-compatible format:
    <run_dir>/successes.hdf5
    <run_dir>/failures.hdf5

Usage:
    python combine_rollouts_to_hdf5.py --run_dir rollout_videos/run_20260313_120000
"""

import os
import argparse
import glob
import numpy as np
import h5py


def build_hdf5(npz_paths, output_path):
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        total_samples = 0

        for demo_idx, npz_path in enumerate(sorted(npz_paths)):
            npz = np.load(npz_path, allow_pickle=True)
            demo_grp = data_grp.create_group(f"demo_{demo_idx}")

            actions = npz["actions"]
            demo_grp.create_dataset("actions", data=actions)
            demo_grp.attrs["num_samples"] = len(actions)
            total_samples += len(actions)

            obs_grp = demo_grp.create_group("obs")
            for key in npz.files:
                if key.startswith("obs_"):
                    obs_key = key[len("obs_"):]
                    obs_grp.create_dataset(obs_key, data=npz[key])

        data_grp.attrs["total"] = total_samples

    print(f"  Wrote {len(npz_paths)} demos ({total_samples} samples) -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directory containing rollout_*.npz files from minimal_eval.py")
    args = parser.parse_args()

    all_npz = glob.glob(os.path.join(args.run_dir, "rollout_*.npz"))
    if not all_npz:
        print(f"No .npz files found in {args.run_dir}")
        return

    successes = [p for p in all_npz if "_success.npz" in p]
    failures  = [p for p in all_npz if "_fail.npz" in p]

    print(f"Found {len(successes)} successes and {len(failures)} failures in {args.run_dir}")

    if successes:
        build_hdf5(successes, os.path.join(args.run_dir, "successes.hdf5"))
    if failures:
        build_hdf5(failures, os.path.join(args.run_dir, "failures.hdf5"))


if __name__ == "__main__":
    main()
