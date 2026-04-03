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
            with torch.no_grad():
                eval_info = evaluator.evaluate(model, seed=args.seed)
            
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
            
            results.append({
                'epoch': ep,
                'success_rate': succ_rate,
                'num_success': num_success,
                'num_fail': num_fail,
                'mean_steps': mean_steps
            })
            
            print(f"> Result for Epoch {ep}: {num_success}/{eval_config.n_rollouts} successes ({succ_rate*100:.1f}%)")
            
            # Save progressively after every epoch in case it crashes
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"> Progress saved to {args.output}")
            
        except Exception as e:
            print(f"> Failed to evaluate epoch {ep}. Error: {e}")
    
    wandb.finish()
    print(f"\nFinal Evaluation Complete. All results successfully saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config (e.g. flow_policy_config.py)")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to the trained weights directory")
    parser.add_argument("--output", type=str, default="baseline_policy.csv", help="Output CSV filename")
    parser.add_argument("--run_name", type=str, default=None, help="Custom wandb run name (defaults to weights_dir basename)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic simulator initial states across checkpoints")
    args = parser.parse_args()
    main(args)
