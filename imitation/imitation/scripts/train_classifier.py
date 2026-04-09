"""
Train a binary success/failure trajectory classifier.

Usage:
    python train_classifier.py \
        --config ./config/model/classifier_config.py \
        --exp_name classifier_v1

Produces:
    - Saved model checkpoints
    - Per-trajectory prediction plots (P(success) over time)
    - Val accuracy, AUC, and critical-point accuracy logged to wandb
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from importlib.machinery import SourceFileLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from imitation.utils.general_utils import AttrDict
from imitation.utils.obs_utils import process_obs_dict
from imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from imitation.utils.file_utils import get_all_obs_keys_from_config, get_shape_metadata_from_dataset, get_obs_key_to_modality_from_config
from imitation.models.trajectory_classifier import TrajectoryClassifier

DEVICE = 'cuda'

LOG = True
WANDB_PROJECT_NAME = 'trajectory-classifier'
WANDB_ENTITY_NAME = 'learning-with-negative-examples'


def build_model_config(conf, normalization_stats, shape_meta, obs_key_to_modality):
    return AttrDict(
        classifier_config=conf.classifier_config,
        observation_config=conf.observation_config,
        keys_to_shapes=shape_meta,
        keys_to_modality=obs_key_to_modality,
        normalization_stats=normalization_stats,
    )


def evaluate(model, val_loader, obs_key_to_modality, criterion):
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch['obs'] = process_obs_dict(batch['obs'], obs_key_to_modality)
            batch = recursive_dict_list_tuple_apply(batch, {torch.Tensor: lambda x: x.to(DEVICE).float()})

            logits = model(batch['obs'], batch['actions']).squeeze(-1)
            labels = batch['label']

            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(float)

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {
        'val_loss': total_loss / max(n_batches, 1),
        'val_accuracy': acc,
        'val_auc': auc,
    }


def plot_trajectory_predictions(model, dataset, obs_key_to_modality, save_dir, epoch, n_demos=6):
    """
    For a few success and failure demos, plot the classifier's P(success)
    at each timestep. Saves matplotlib figures.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping trajectory plots")
        return

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    demos_to_plot = []
    n_files = len(dataset.hdf5_paths)
    for file_idx in range(n_files):
        cache = dataset.cache[file_idx]
        demo_keys = sorted(cache.keys(), key=lambda x: int(x.split('_')[-1]))
        label = 1.0 if file_idx < dataset.num_pos else 0.0
        count = 0
        for dk in demo_keys:
            if count >= n_demos // n_files:
                break
            demos_to_plot.append((file_idx, dk, label))
            count += 1

    for file_idx, demo_key, label in demos_to_plot:
        traj_data = dataset.get_trajectory_data(file_idx, demo_key)
        n_windows = traj_data['actions'].shape[0]

        batch_size = 128
        all_probs = []
        with torch.no_grad():
            for start in range(0, n_windows, batch_size):
                end = min(start + batch_size, n_windows)
                obs_batch = {k: traj_data['obs'][k][start:end] for k in traj_data['obs']}
                act_batch = traj_data['actions'][start:end]

                obs_batch = process_obs_dict(obs_batch, obs_key_to_modality)
                obs_batch = recursive_dict_list_tuple_apply(
                    obs_batch, {np.ndarray: lambda x: torch.from_numpy(x).float().to(DEVICE)}
                )
                act_tensor = torch.from_numpy(act_batch).float().to(DEVICE)

                probs = model.predict_prob(obs_batch, act_tensor).squeeze(-1).cpu().numpy()
                all_probs.extend(probs)

        timesteps = np.arange(len(all_probs))
        tag = "success" if label > 0.5 else "failure"

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(timesteps, all_probs, linewidth=1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('P(success)')
        ax.set_title(f'{tag} | file={file_idx} demo={demo_key} | len={traj_data["demo_len"]}')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'ep{epoch}_{tag}_{demo_key}.png'), dpi=100)
        plt.close(fig)


def main(args):
    conf = SourceFileLoader('conf', args.config).load_module().config
    train_config = conf.train_config
    data_config = conf.data_config

    obs_keys = get_all_obs_keys_from_config(conf.observation_config)
    obs_key_to_modality = get_obs_key_to_modality_from_config(conf.observation_config)
    shape_meta = get_shape_metadata_from_dataset(data_config.data[0], all_obs_keys=obs_keys, obs_key_to_modality=obs_key_to_modality)

    print("Building datasets...")
    ds_kwargs = dict(data_config.dataset_kwargs)
    ds_kwargs['obs_keys_to_modality'] = obs_key_to_modality
    ds_kwargs['obs_keys_to_normalize'] = conf.observation_config.obs_keys_to_normalize

    train_dataset = data_config.dataset_class(data_paths=data_config.data, split='train', **ds_kwargs)
    val_dataset = data_config.dataset_class(data_paths=data_config.data, split='val', **ds_kwargs)

    train_sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, sampler=train_sampler, num_workers=train_config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=train_config.num_workers)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    normalization_stats = train_dataset.get_normalization_stats()
    model_config = build_model_config(conf, normalization_stats, shape_meta, obs_key_to_modality)
    model = TrajectoryClassifier(model_config).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Classifier parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    output_dir = os.path.expanduser(train_config.output_dir)
    exp_dir = os.path.join(output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    plot_dir = os.path.join(exp_dir, 'trajectory_plots')

    logger = None
    if LOG:
        try:
            from imitation.utils.log_utils import WandBLogger
            logger = WandBLogger(args.exp_name, WANDB_PROJECT_NAME, WANDB_ENTITY_NAME, exp_dir, conf)
        except Exception as e:
            print(f"WandB logging disabled: {e}")

    best_auc = 0.0

    for epoch in range(train_config.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{train_config.num_epochs}"):
            batch['obs'] = process_obs_dict(batch['obs'], obs_key_to_modality)
            batch = recursive_dict_list_tuple_apply(batch, {torch.Tensor: lambda x: x.to(DEVICE).float()})

            logits = model(batch['obs'], batch['actions']).squeeze(-1)
            loss = criterion(logits, batch['label'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch} | Train loss: {avg_loss:.4f}")

        if logger:
            logger.log_scalar_dict({'loss': avg_loss}, step=epoch, phase='train')

        if (epoch + 1) % train_config.val_every_n_epochs == 0:
            val_metrics = evaluate(model, val_loader, obs_key_to_modality, criterion)
            print(f"  Val loss: {val_metrics['val_loss']:.4f} | "
                  f"Acc: {val_metrics['val_accuracy']:.3f} | "
                  f"AUC: {val_metrics['val_auc']:.3f}")

            if logger:
                logger.log_scalar_dict(val_metrics, step=epoch, phase='val')

            if val_metrics['val_auc'] > best_auc:
                best_auc = val_metrics['val_auc']
                model.save(os.path.join(exp_dir, 'best_classifier.pth'))
                print(f"  New best AUC: {best_auc:.3f}")

            plot_trajectory_predictions(
                model, val_dataset, obs_key_to_modality, plot_dir, epoch,
                n_demos=6
            )

        if (epoch + 1) % train_config.save_every_n_epochs == 0:
            model.save(os.path.join(exp_dir, f'classifier_ep{epoch+1}.pth'))

    model.save(os.path.join(exp_dir, 'classifier_final.pth'))
    print(f"\nTraining complete. Best val AUC: {best_auc:.3f}")
    print(f"Models saved to: {exp_dir}")
    print(f"Trajectory plots saved to: {plot_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to classifier config")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    args = parser.parse_args()
    main(args)
