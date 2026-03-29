"""
Inspect HDF5 demo file to see what data is available per timestep,
especially checking for reward signals.

Usage:
    python inspect_hdf5.py --path ./data/success.hdf5
    python inspect_hdf5.py --path ./data/failure.hdf5 --demo demo_0
"""
import h5py
import numpy as np
import argparse


def inspect_hdf5(path, demo_key=None):
    with h5py.File(path, 'r') as f:
        print(f"=== File: {path} ===\n")

        # Top-level keys
        print(f"Top-level keys: {list(f.keys())}")
        if 'data' not in f:
            print("ERROR: No 'data' group found in file.")
            return

        demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
        print(f"Number of demos: {len(demos)}")
        print(f"Demo keys: {demos[:5]}{'...' if len(demos) > 5 else ''}\n")

        # Pick a demo to inspect
        if demo_key is None:
            demo_key = demos[0]
        
        demo = f[f'data/{demo_key}']
        print(f"=== Inspecting {demo_key} ===")
        print(f"Keys in demo: {list(demo.keys())}")

        # Print attributes
        if demo.attrs:
            print(f"Attributes: {dict(demo.attrs)}")
        print()

        # Print shape of each dataset
        def print_tree(group, prefix=""):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"  {prefix}{key}: shape={item.shape}, dtype={item.dtype}")
                elif isinstance(item, h5py.Group):
                    print(f"  {prefix}{key}/")
                    print_tree(item, prefix=prefix + "  ")

        print_tree(demo)
        print()

        # Check for reward-like keys
        reward_keys = [k for k in demo.keys() if 'reward' in k.lower()]
        if reward_keys:
            print(f"=== REWARD DATA FOUND: {reward_keys} ===")
            for rk in reward_keys:
                rewards = demo[rk][:]
                print(f"\n  {rk}: shape={rewards.shape}")
                print(f"  min={rewards.min():.6f}, max={rewards.max():.6f}, mean={rewards.mean():.6f}")
                print(f"  First 20 values: {rewards[:20].flatten()}")
                print(f"  Last 20 values:  {rewards[-20:].flatten()}")
                
                # Cumulative reward
                cumulative = np.cumsum(rewards.flatten())
                peak_idx = np.argmax(cumulative)
                print(f"  Cumulative reward peak at timestep {peak_idx}/{len(rewards)}")
                print(f"  Cumulative reward at peak: {cumulative[peak_idx]:.4f}")
        else:
            print("=== NO REWARD DATA FOUND ===")
            print("Available keys:", list(demo.keys()))

        # Check for done/success signals
        for key in ['dones', 'done', 'success', 'terminals']:
            if key in demo:
                data = demo[key][:]
                print(f"\n  {key}: shape={data.shape}, unique values={np.unique(data)}")

        # Also check if rewards are stored in env_infos or similar
        for key in demo.keys():
            if isinstance(demo[key], h5py.Group):
                sub_keys = list(demo[key].keys())
                reward_sub = [k for k in sub_keys if 'reward' in k.lower() or 'success' in k.lower()]
                if reward_sub:
                    print(f"\n=== Found reward-related keys in {key}/: {reward_sub} ===")
                    for rk in reward_sub:
                        data = demo[f'{key}/{rk}'][:]
                        print(f"  {key}/{rk}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
                        print(f"  First 20: {data[:20].flatten()}")

        # Print actions info
        if 'actions' in demo:
            actions = demo['actions'][:]
            print(f"\n=== Actions ===")
            print(f"  shape={actions.shape} (timesteps={actions.shape[0]}, action_dim={actions.shape[1]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to HDF5 file')
    parser.add_argument('--demo', type=str, default=None, help='Demo key to inspect (default: first demo)')
    args = parser.parse_args()

    inspect_hdf5(args.path, args.demo)
