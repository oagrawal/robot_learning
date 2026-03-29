"""
Label every timestep in failure demos as positive (c=1.0) or negative (c=0.0)
based on the max(hover) from staged_rewards.

Strategy:
  - For each failure demo, find the timestep where hover is maximized
  - This is the "peak progress" point — the closest the nut got to the peg
  - Timesteps up to (peak - n_buffer) → c = 1.0 (robot was making progress)
  - Timesteps after (peak - n_buffer) → c = 0.0 (robot failed / regressed)
  
  For demos where hover never exceeds baseline (~0.001, i.e. never grasped),
  all timesteps are labeled c = 0.0.

Usage:
    # Dry run to see statistics
    python label_timesteps.py --input ./failure_dense.hdf5 --dry_run

    # Label with default buffer (8 = action horizon)
    python label_timesteps.py --input ./failure_dense.hdf5 --output ./failure_labeled.hdf5

    # Label with no buffer
    python label_timesteps.py --input ./failure_dense.hdf5 --output ./failure_labeled.hdf5 --n_buffer 0
"""

import os
import argparse
import shutil
import numpy as np
import h5py
from tqdm import tqdm


def label_timesteps(input_path, output_path, min_hover_threshold=0.1, n_buffer=8, dry_run=False):
    """
    Post-process a failure HDF5 file to add per-timestep c labels using max(hover).
    
    Args:
        input_path: Path to HDF5 with staged_rewards (from collect_rollouts.py)
        output_path: Path to write labeled HDF5
        min_hover_threshold: Minimum peak hover for the demo to have any positive labels.
                             If peak hover never exceeds this, all timesteps are negative.
                             Default 0.1 (just above reaching-only levels ~0.0004).
        n_buffer: Number of timesteps before the hover peak to also label as failure.
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
            'all_negative_demos': 0,
            'divergence_points': [],
            'peak_hovers': [],
            'demo_lengths': [],
        }
        
        labels_per_demo = {}
        
        for demo_key in tqdm(demos, desc="Analyzing demos"):
            demo = f[f'data/{demo_key}']
            n_steps = demo['actions'].shape[0]
            
            stats['total_timesteps'] += n_steps
            stats['demo_lengths'].append(n_steps)
            
            # Get hover values from staged_rewards (column index 3)
            if 'staged_rewards' in demo:
                staged = demo['staged_rewards'][:]
                hover = staged[:, 3]  # (reach, grasp, lift, hover)
            else:
                # Fallback: use scalar reward
                hover = demo['rewards'][:]
            
            # Find peak hover (peak progress toward peg)
            peak_idx = np.argmax(hover)
            peak_hover = hover[peak_idx]
            
            stats['peak_hovers'].append(peak_hover)
            
            # Generate labels
            c_labels = np.zeros(n_steps, dtype=np.float32)
            
            if peak_hover >= min_hover_threshold:
                # Robot made meaningful progress — label pre-peak as positive
                cutoff = max(0, peak_idx - n_buffer + 1)
                c_labels[:cutoff] = 1.0
                c_labels[cutoff:] = 0.0
                stats['divergence_points'].append(cutoff)
                stats['positive_timesteps'] += cutoff
                stats['negative_timesteps'] += (n_steps - cutoff)
            else:
                # Robot never made meaningful progress (never grasped) — all negative
                c_labels[:] = 0.0
                stats['all_negative_demos'] += 1
                stats['negative_timesteps'] += n_steps
            
            labels_per_demo[demo_key] = c_labels
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"LABELING STATISTICS (max hover approach)")
        print(f"{'='*60}")
        print(f"Total demos:           {len(demos)}")
        print(f"Total timesteps:       {stats['total_timesteps']}")
        print(f"Positive timesteps:    {stats['positive_timesteps']} ({stats['positive_timesteps']/stats['total_timesteps']:.1%})")
        print(f"Negative timesteps:    {stats['negative_timesteps']} ({stats['negative_timesteps']/stats['total_timesteps']:.1%})")
        print(f"All-negative demos:    {stats['all_negative_demos']} (never grasped / never made progress)")
        print(f"Buffer (n_buffer):     {n_buffer} timesteps before peak also labeled negative")
        
        if stats['divergence_points']:
            div_pts = np.array(stats['divergence_points'])
            lengths = np.array(stats['demo_lengths'])[:len(div_pts)]
            pcts = div_pts / lengths * 100
            print(f"\nDivergence point stats (for demos with progress):")
            print(f"  Mean:    timestep {div_pts.mean():.0f} ({pcts.mean():.0f}% through demo)")
            print(f"  Median:  timestep {np.median(div_pts):.0f} ({np.median(pcts):.0f}% through demo)")
            print(f"  Min:     timestep {div_pts.min()} ({pcts.min():.0f}%)")
            print(f"  Max:     timestep {div_pts.max()} ({pcts.max():.0f}%)")
        
        peak_hovs = np.array(stats['peak_hovers'])
        print(f"\nPeak hover stats:")
        print(f"  Mean: {peak_hovs.mean():.4f}")
        print(f"  Min:  {peak_hovs.min():.4f}")
        print(f"  Max:  {peak_hovs.max():.4f}")
        print(f"  Demos with hover > 0.5 (lifted):  {np.sum(peak_hovs > 0.5)}")
        print(f"  Demos with hover > 0.35 (grasped): {np.sum(peak_hovs > 0.35)}")
        print(f"  Demos with hover < {min_hover_threshold} (no progress): {np.sum(peak_hovs < min_hover_threshold)}")
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
            
            if 'c_labels' in demo:
                del demo['c_labels']
            demo.create_dataset('c_labels', data=labels_per_demo[demo_key])
    
    print(f"Done! Labeled dataset saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input HDF5 with staged_rewards')
    parser.add_argument('--output', type=str, default=None, help='Output HDF5 path (default: overwrites input)')
    parser.add_argument('--min_hover', type=float, default=0.1, help='Min peak hover for positive labels (default: 0.1)')
    parser.add_argument('--n_buffer', type=int, default=8, help='Steps before hover peak to also label as failure (default: 8)')
    parser.add_argument('--dry_run', action='store_true', help='Only print stats, do not write output')
    args = parser.parse_args()
    
    output = args.output if args.output else args.input
    label_timesteps(args.input, output, args.min_hover, args.n_buffer, args.dry_run)
