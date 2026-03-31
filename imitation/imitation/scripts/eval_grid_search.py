import os
import argparse
import itertools
import pandas as pd
from importlib.machinery import SourceFileLoader
import torch

from imitation.algo.base_algo import BaseAlgo

def main(args):
    # Load config to get evaluator config
    conf = SourceFileLoader('conf', args.config).load_module().config
    eval_config = conf.evaluator_config
    
    # Disable video and npz saving for grid search to save time & space
    eval_config.save_video = False
    eval_config.save_npz = False
    eval_config.n_rollouts = 30  # As you requested
    
    # Load model
    print(f"Loading base model structure from {args.checkpoint}")
    try:
        model = BaseAlgo.load_weights(args.checkpoint)
    except ValueError:
        raise ValueError(
            "\n\nERROR: You provided 'best_val_model.pth' or 'best_eval_model.pth' which only contains raw weights (not the model architecture config)."
            "\n\nTo evaluate this model, provide ANY standard epoch checkpoint as the base structure, and pass this file as the state_dict:"
            "\nExample: python eval_grid_search.py --config ... --checkpoint weights/weights_ep500.pth --state_dict best_val_model.pth\n"
        )
    
    if not model:
        raise FileNotFoundError(f"Could not load weights from {args.checkpoint}")
        
    if args.state_dict:
        print(f"Overriding weights with state_dict from: {args.state_dict}")
        state = torch.load(args.state_dict, map_location='cpu')
        model.load_state_dict(state)
        
    model.to('cuda')
    model.eval()
    
    # Initialize the evaluator
    evaluator = eval_config.evaluator(eval_config=eval_config)
    
    # Define grid.
    # Choose reasonable ranges that avoid Unconditional Cancellation and test Rescale functionality.
    w_succ_values = [1.0, 1.2, 1.5]
    w_fail_values = [0.0, 0.1, 0.5]
    rescale_phi_values = [0.5, 0.7, 1.0]
    
    results = []
    
    total_configs = len(w_succ_values) * len(w_fail_values) * len(rescale_phi_values)
    print(f"\nStarting Grid Search. Total configurations to evaluate: {total_configs}")
    
    for w_s, w_f, phi in itertools.product(w_succ_values, w_fail_values, rescale_phi_values):
        # Skip configurations that cause mathematical cancellation (w_succ - w_fail == 1.0)
        # except for the pure conditional baseline (1.0 - 0.0)
        if abs((w_s - w_f) - 1.0) < 1e-5 and w_s != 1.0:
            print(f"\nSkipping functionally broken config: w_succ={w_s:.1f}, w_fail={w_f:.1f} (Uncond Cancellation)")
            continue

        print(f"\n{'-'*55}")
        print(f"Evaluating --- w_succ = {w_s:.1f} | w_fail = {w_f:.1f} | rescale_phi = {phi:.1f}")
        print(f"{'-'*55}")
        
        # Override the policy's weights dynamically
        model.w_succ = w_s
        model.w_fail = w_f
        model.rescale_phi = phi
        
        # Run evaluation (rollouts)
        with torch.no_grad():
            eval_info = evaluator.evaluate(model)
        
        succ_rate = eval_info['success_rate']
        mean_steps = eval_info['mean_timesteps']
        
        num_success = int(succ_rate * eval_config.n_rollouts)
        num_fail = eval_config.n_rollouts - num_success
        
        results.append({
            'w_succ': w_s,
            'w_fail': w_f,
            'rescale_phi': phi,
            'success_rate': succ_rate,
            'num_success': num_success,
            'num_fail': num_fail,
            'mean_steps': mean_steps
        })
        
        print(f"> Result: {num_success}/{eval_config.n_rollouts} successes ({succ_rate*100:.1f}%)")
        
    df = pd.DataFrame(results)
    
    # Save results to a CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print(f"\n{'='*40}")
    print(f"Grid search complete. Results saved to {args.output}")
    print(f"{'='*40}")
    
    # Display the table via pandas
    print(df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file (e.g. cfg_policy_config.py)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a standard weights_epN.pth file to build the architecture")
    parser.add_argument("--state_dict", type=str, default=None, help="Optional: Path to a raw state_dict (like best_val_model.pth) to override the weights.")
    parser.add_argument("--output", type=str, default="grid_search_results.csv", help="Where to save the CSV results")
    args = parser.parse_args()
    main(args)
