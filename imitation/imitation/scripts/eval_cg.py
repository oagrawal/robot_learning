"""
Evaluate Classifier Guidance (CG) over a grid of alpha values.

Usage:
    python eval_cg.py \
        --config ./config/model/cg_policy_config.py \
        --output cg_results.csv \
        --seed 42
"""

import os
import argparse
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader

from imitation.algo.base_algo import BaseAlgo
from imitation.algo.cg_policy import CGPolicy
from imitation.models.trajectory_classifier import TrajectoryClassifier


def main(args):
    conf = SourceFileLoader('conf', args.config).load_module().config
    cg_config = conf.cg_config
    eval_config = conf.evaluator_config

    eval_config.save_video = False
    eval_config.save_npz = False

    base_policy_ckpt = os.path.expanduser(cg_config.base_policy_ckpt)
    classifier_ckpt = os.path.expanduser(cg_config.classifier_ckpt)

    print(f"Loading base policy from: {base_policy_ckpt}")
    base_policy = BaseAlgo.load_weights(base_policy_ckpt)
    if not base_policy:
        raise FileNotFoundError(f"Could not load base policy from {base_policy_ckpt}")
    base_policy.to('cuda')
    base_policy.eval()

    print(f"Loading classifier from: {classifier_ckpt}")
    classifier = TrajectoryClassifier.load(classifier_ckpt, device='cuda')
    classifier.to('cuda')
    classifier.eval()

    evaluator = eval_config.evaluator(eval_config=eval_config)

    alpha_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    results = []
    print(f"\nStarting CG evaluation. Alpha values: {alpha_values}")
    print(f"Rollouts per config: {eval_config.n_rollouts}")

    for alpha in alpha_values:
        print(f"\n{'='*55}")
        print(f"Evaluating alpha = {alpha}")
        print(f"{'='*55}")

        cg_policy = CGPolicy(base_policy, classifier, alpha=alpha)
        cg_policy.to('cuda')
        cg_policy.eval()

        eval_info = evaluator.evaluate(cg_policy, seed=args.seed)

        succ_rate = eval_info['success_rate']
        mean_steps = eval_info['mean_timesteps']
        num_success = int(succ_rate * eval_config.n_rollouts)

        results.append({
            'alpha': alpha,
            'success_rate': succ_rate,
            'num_success': num_success,
            'num_fail': eval_config.n_rollouts - num_success,
            'mean_steps': mean_steps,
        })

        print(f"> Result: {num_success}/{eval_config.n_rollouts} successes ({succ_rate*100:.1f}%)")

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*55}")
    print("CG evaluation complete.")
    print(f"{'='*55}")
    print(df.to_string(index=False))
    print(f"\nResults saved to: {args.output}")

    best = df.loc[df['success_rate'].idxmax()]
    print(f"\nBest alpha: {best['alpha']} with success rate: {best['success_rate']*100:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to cg_policy_config.py")
    parser.add_argument("--output", type=str, default="cg_results.csv",
                        help="Where to save CSV results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed RNG seed for reproducible initial states")
    args = parser.parse_args()
    main(args)
