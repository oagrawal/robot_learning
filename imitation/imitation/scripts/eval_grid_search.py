import os
import argparse
import itertools
import pandas as pd
from importlib.machinery import SourceFileLoader

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
    # We can sweep over a sensible range for both weights.
    w_succ_values = [1.0, 1.5, 2.0, 2.5]
    w_fail_values = [0.0, 0.5, 1.0, 1.5]
    
    results = []
    
    print(f"\nStarting Grid Search. Total configurations to evaluate: {len(w_succ_values) * len(w_fail_values)}")
    
    for w_s, w_f in itertools.product(w_succ_values, w_fail_values):
        print(f"\n{'-'*40}")
        print(f"Evaluating --- w_succ = {w_s:.1f} | w_fail = {w_f:.1f}")
        print(f"{'-'*40}")
        
        # Override the policy's weights dynamically
        model.w_succ = w_s
        model.w_fail = w_f
        
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
