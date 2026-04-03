import os
import argparse
import pandas as pd
from importlib.machinery import SourceFileLoader
import torch
import re
import numpy as np
import wandb

from imitation.algo.base_algo import BaseAlgo

WANDB_PROJECT_NAME = 'positive-flow-policy-evals'
WANDB_ENTITY_NAME = 'learning-with-negative-examples'

def main(args):
    # Load config to get evaluator config
    print(f"Loading config from: {args.config}")
    conf = SourceFileLoader('conf', args.config).load_module().config
    eval_config = conf.evaluator_config
    
    # Override settings for this specific evaluation run
    eval_config.save_video = False
    eval_config.save_npz = False
    eval_config.n_rollouts = 100  # Do exactly 100 rollouts
    
    # Initialize the evaluator
    evaluator = eval_config.evaluator(eval_config=eval_config)
    
    # Initialize wandb in a separate eval-only project
    run_name = args.run_name or os.path.basename(os.path.normpath(args.weights_dir))
    wandb.init(
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY_NAME,
        name=run_name,
        config={
            'config_path': args.config,
            'weights_dir': args.weights_dir,
            'n_rollouts': eval_config.n_rollouts,
            'seed': args.seed,
        }
    )
    
    results = []
    init_hashes = []  # collect per-checkpoint init-state hashes for comparison
    
    # 1. Find and sort all checkpoint paths (filtered for multiples of 50)
    print(f"Scanning directory: {args.weights_dir}")
    checkpoint_files = []
    
    if not os.path.exists(args.weights_dir):
        raise NotADirectoryError(f"Directory not found: {args.weights_dir}")
        
    for f in os.listdir(args.weights_dir):
        match = re.search(r'weights_ep(\d+)\.pth', f)
        if match:
            ep = int(match.group(1))
            # The prompt requested "at every 50 epochs"
            if ep % 50 == 0:
                checkpoint_files.append((ep, os.path.join(args.weights_dir, f)))
                
    checkpoint_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(checkpoint_files)} checkpoints to evaluate.")
    
    # 2. Iterate and evaluate
    for ep, ckpt_path in checkpoint_files:
        print(f"\n{'='*50}")
        print(f"Evaluating Epoch {ep} (File: {os.path.basename(ckpt_path)})")
        print(f"{'='*50}")
        
        try:
            model = BaseAlgo.load_weights(ckpt_path)
            model.to('cuda')
            model.eval()
            
            # If evaluating the CFG policy model, make sure it evaluates purely unconditioned 
            # or purely success-conditioned (uncomment if you want to test pure success)
            if hasattr(model, 'w_succ'):
                model.w_succ = 1.0
                model.w_fail = 0.0
                model.rescale_phi = 0.0

            # Run the rollouts — use a fixed seed so every checkpoint
            # starts the simulator in the same state for each rollout.
            # Save first-frame images for the first 2 checkpoints so the
            # user can visually compare that initial states match.
            verif_dir = None
            if args.verification_frames_dir and len(results) < 2:
                verif_dir = args.verification_frames_dir
            
            with torch.no_grad():
                eval_info = evaluator.evaluate(
                    model, epoch=ep, seed=args.seed,
                    verification_frames_dir=verif_dir,
                )
            
            succ_rate = eval_info['success_rate']
            fail_rate = 1.0 - succ_rate
            mean_steps = eval_info['mean_timesteps']
            num_success = int(succ_rate * eval_config.n_rollouts)
            num_fail = eval_config.n_rollouts - num_success
            
            # Log to wandb
            wandb.log({
                'success_rate': succ_rate,
                'failure_rate': fail_rate,
            }, step=ep)
            
            ckpt_hash = eval_info.get('init_state_hash', None)
            init_hashes.append((ep, ckpt_hash))
            
            results.append({
                'epoch': ep,
                'success_rate': succ_rate,
                'num_success': num_success,
                'num_fail': num_fail,
                'mean_steps': mean_steps,
                'init_state_hash': ckpt_hash,
            })
            
            print(f"> Result for Epoch {ep}: {num_success}/{eval_config.n_rollouts} successes ({succ_rate*100:.1f}%)")
            
            # Save progressively after every epoch in case it crashes
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"> Progress saved to {args.output}")
            
        except Exception as e:
            print(f"> Failed to evaluate epoch {ep}. Error: {e}")
    
    # --- Cross-checkpoint seed verification ---
    if init_hashes:
        unique_hashes = set(h for _, h in init_hashes if h is not None)
        print(f"\n{'='*50}")
        print("SEED VERIFICATION: Cross-checkpoint init-state comparison")
        print(f"{'='*50}")
        for ep, h in init_hashes:
            print(f"  Epoch {ep:4d}: {h}")
        if len(unique_hashes) == 1:
            print("\n  ✓ ALL CHECKPOINTS HAVE IDENTICAL INITIAL STATES")
        else:
            print(f"\n  ✗ WARNING: Found {len(unique_hashes)} different init-state hashes!")
            print("    Initial states are NOT identical across checkpoints.")
        print(f"{'='*50}\n")
    
    wandb.finish()
    print(f"\nFinal Evaluation Complete. All results successfully saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config (e.g. flow_policy_config.py)")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to the trained weights directory")
    parser.add_argument("--output", type=str, default="baseline_policy.csv", help="Output CSV filename")
    parser.add_argument("--run_name", type=str, default=None, help="Custom wandb run name (defaults to weights_dir basename)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic simulator initial states across checkpoints")
    parser.add_argument("--verification_frames_dir", type=str, default="verification_frames",
                        help="Directory for first-frame verification images (first 2 checkpoints, 3 rollouts each). Set to empty string to disable.")
    args = parser.parse_args()
    main(args)
