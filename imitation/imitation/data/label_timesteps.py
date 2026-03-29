"""
Label every timestep in a failure HDF5 file as positive (c=1.0) or negative (c=0.0)
based on the dense shaped reward signal.

Strategy:
  - Track the cumulative shaped reward over the trajectory
  - Find the peak of the cumulative reward (last point of forward progress)
  - Timesteps up to and including the peak → c = 1.0 (robot was making progress)
  - Timesteps after the peak → c = 0.0 (robot failed / regressed)
  
  For demos where reward never exceeds a minimum threshold (e.g., never grasped),
  we label all timesteps as c = 0.0 since the robot never did the right thing.

Usage:
    python label_timesteps.py --input ./data/failure_dense.hdf5 --output ./data/failure_labeled.hdf5
    python label_timesteps.py --input ./data/failure_dense.hdf5 --output ./data/failure_labeled.hdf5 --min_reward 0.05 --n_buffer 8
    
    # Dry run first to see statistics:
    python label_timesteps.py --input ./data/failure_dense.hdf5 --dry_run
"""

import os
import argparse
import shutil
import numpy as np
import h5py
from tqdm import tqdm


def label_timesteps(input_path, output_path, min_reward_threshold=0.05, n_buffer=8, dry_run=False):
    """
    Post-process a failure HDF5 file to add per-timestep c labels.
    
    Args:
        input_path: Path to HDF5 with dense rewards (from collect_failures.py)
        output_path: Path to write labeled HDF5
        min_reward_threshold: Minimum peak reward for the demo to have any positive labels.
                             If the peak reward never exceeds this, all timesteps are negative.
        n_buffer: Number of timesteps before the reward peak to also label as failure.
                  Captures the causal actions that led to the failure. Default 8 (action horizon).
        dry_run: If True, only print statistics without writing output.
    """
    
    with h5py.File(input_path, 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
        print(f"Found {len(demos)} demos in {input_path}\n")
        
        stats = {
            'total_timesteps': 0,
            'positive_timesteps': 0,
            'negative_timesteps': 0,
            'all_negative_demos': 0,     # demos where robot never made progress
            'divergence_points': [],      # where failure happened in each demo 
            'peak_rewards': [],
            'demo_lengths': [],
        }
        
        labels_per_demo = {}
        
        for demo_key in tqdm(demos, desc="Analyzing demos"):
            demo = f[f'data/{demo_key}']
            rewards = demo['rewards'][:]
            n_steps = len(rewards)
            
            stats['total_timesteps'] += n_steps
            stats['demo_lengths'].append(n_steps)
            
            # Compute cumulative reward to find peak progress
            cumulative = np.cumsum(rewards)
            peak_idx = np.argmax(cumulative)
            peak_reward = cumulative[peak_idx]
            
            stats['peak_rewards'].append(peak_reward)
            
            # Generate labels
            c_labels = np.zeros(n_steps, dtype=np.float32)
            
            if peak_reward >= min_reward_threshold:
                # Robot made some progress — label pre-peak as positive, with buffer
                cutoff = max(0, peak_idx - n_buffer + 1)  # buffer eats into positive region
                c_labels[:cutoff] = 1.0
                c_labels[cutoff:] = 0.0
                stats['divergence_points'].append(cutoff)
                stats['positive_timesteps'] += cutoff
                stats['negative_timesteps'] += (n_steps - cutoff)
            else:
                # Robot never made meaningful progress — all negative
                c_labels[:] = 0.0
                stats['all_negative_demos'] += 1
                stats['negative_timesteps'] += n_steps
            
            labels_per_demo[demo_key] = c_labels
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"LABELING STATISTICS")
        print(f"{'='*60}")
        print(f"Total demos:           {len(demos)}")
        print(f"Total timesteps:       {stats['total_timesteps']}")
        print(f"Positive timesteps:    {stats['positive_timesteps']} ({stats['positive_timesteps']/stats['total_timesteps']:.1%})")
        print(f"Negative timesteps:    {stats['negative_timesteps']} ({stats['negative_timesteps']/stats['total_timesteps']:.1%})")
        print(f"All-negative demos:    {stats['all_negative_demos']} (never made progress)")
        
        if stats['divergence_points']:
            div_pts = np.array(stats['divergence_points'])
            lengths = np.array(stats['demo_lengths'])[:len(div_pts)]
            pcts = div_pts / lengths * 100
            print(f"\nDivergence point stats (for demos with progress):")
            print(f"  Mean:    timestep {div_pts.mean():.0f} ({pcts.mean():.0f}% through demo)")
            print(f"  Median:  timestep {np.median(div_pts):.0f} ({np.median(pcts):.0f}% through demo)")
            print(f"  Min:     timestep {div_pts.min()} ({pcts.min():.0f}%)")
            print(f"  Max:     timestep {div_pts.max()} ({pcts.max():.0f}%)")
        
        peak_rews = np.array(stats['peak_rewards'])
        print(f"\nPeak cumulative reward stats:")
        print(f"  Mean: {peak_rews.mean():.4f}")
        print(f"  Min:  {peak_rews.min():.4f}")
        print(f"  Max:  {peak_rews.max():.4f}")
        print(f"{'='*60}\n")
        
        if dry_run:
            print("DRY RUN — no output written.")
            return
    
    # Write the labeled HDF5
    print(f"Writing labeled dataset to {output_path}")
    shutil.copy2(input_path, output_path)
    
    with h5py.File(output_path, 'a') as f:
        for demo_key in tqdm(demos, desc="Writing labels"):
            demo = f[f'data/{demo_key}']
            
            # Add per-timestep c labels
            if 'c_labels' in demo:
                del demo['c_labels']
            demo.create_dataset('c_labels', data=labels_per_demo[demo_key])
    
    print(f"Done! Labeled dataset saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input HDF5 with dense rewards')
    parser.add_argument('--output', type=str, default=None, help='Output HDF5 path (default: overwrites input)')
    parser.add_argument('--min_reward', type=float, default=0.05, help='Min peak reward for positive labels')
    parser.add_argument('--n_buffer', type=int, default=8, help='Steps before reward peak to also label as failure (default: 8 = action horizon)')
    parser.add_argument('--dry_run', action='store_true', help='Only print stats, do not write output')
    args = parser.parse_args()
    
    output = args.output if args.output else args.input
    label_timesteps(args.input, output, args.min_reward, args.n_buffer, args.dry_run)
