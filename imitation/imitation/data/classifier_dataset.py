import os
import h5py
import numpy as np
import torch.utils.data
from tqdm import tqdm

from imitation.utils.general_utils import AttrDict
from imitation.utils.obs_utils import process_obs_dict


class ClassifierDataset(torch.utils.data.Dataset):
    """
    Dataset for training a success/failure trajectory classifier.

    Each sample is a window of (obs, action) pairs from a single demo,
    labeled by the demo's outcome (1.0 = success, 0.0 = failure).

    Args:
        data_paths: List of HDF5 paths. First num_pos are success, rest failure.
        num_pos: Number of positive (success) data paths.
        window_size: Number of consecutive timesteps per sample.
        obs_keys_to_modality: Dict mapping obs key names to modality type.
        split: 'train' or 'val'.
    """
    SPLIT = AttrDict(train=0.95, val=0.05)

    def __init__(self, data_paths, num_pos=1, window_size=3,
                 obs_keys_to_modality={}, obs_keys_to_normalize={},
                 split='train', **kwargs):
        super().__init__()

        assert isinstance(data_paths, list)
        self.hdf5_paths = [os.path.expanduser(p) for p in data_paths]
        self.obs_keys = tuple(obs_keys_to_modality.keys())
        self.obs_keys_to_modality = obs_keys_to_modality
        self.window_size = window_size
        self.split = split
        self.num_pos = num_pos

        self._build_index()
        self._cache_low_dim()
        self._compute_normalization_stats(list(obs_keys_to_normalize.keys()))

    def _build_index(self):
        """Build a flat index: each entry is (file_idx, demo_key, timestep, label)."""
        self.index = []
        self.hdf5_use_swmr = True

        for file_idx, path in enumerate(self.hdf5_paths):
            label = 1.0 if file_idx < self.num_pos else 0.0
            with h5py.File(path, 'r', swmr=True, libver='latest') as f:
                demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
                for demo_key in demos:
                    demo_len = f[f'data/{demo_key}/actions'].shape[0]
                    num_windows = max(0, demo_len - self.window_size + 1)
                    for t in range(num_windows):
                        self.index.append((file_idx, demo_key, t, label))

        total = len(self.index)
        self.train_split = int(self.SPLIT.train * total)

        if self.split == 'train':
            self.index = self.index[:self.train_split]
        elif self.split == 'val':
            self.index = self.index[self.train_split:]

    def _cache_low_dim(self):
        """Cache low-dim obs and actions in memory for fast access."""
        self.cache = {}
        for file_idx, path in enumerate(self.hdf5_paths):
            print(f"ClassifierDataset: caching low-dim from {path}")
            self.cache[file_idx] = {}
            with h5py.File(path, 'r', swmr=True, libver='latest') as f:
                demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
                for demo_key in tqdm(demos):
                    demo = f[f'data/{demo_key}']
                    self.cache[file_idx][demo_key] = {
                        'actions': demo['actions'][:].astype(np.float32),
                    }
                    for k in self.obs_keys:
                        if self.obs_keys_to_modality.get(k) == 'low_dim':
                            self.cache[file_idx][demo_key][f'obs/{k}'] = \
                                demo[f'obs/{k}'][:].astype(np.float32)

        self._hdf5_files = None

    @property
    def hdf5_files(self):
        if self._hdf5_files is None:
            self._hdf5_files = [
                h5py.File(p, 'r', swmr=True, libver='latest') for p in self.hdf5_paths
            ]
        return self._hdf5_files

    def _compute_normalization_stats(self, obs_keys_to_normalize):
        """Compute normalization stats from positive (success) data only."""
        merged = None
        for file_idx in range(self.num_pos):
            path = self.hdf5_paths[file_idx]
            with h5py.File(path, 'r', swmr=True, libver='latest') as f:
                demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
                for demo_key in demos:
                    obs_traj = {}
                    for k in obs_keys_to_normalize:
                        arr = f[f'data/{demo_key}/obs/{k}'][()].astype(np.float32)
                        arr = process_obs_dict({k: arr}, self.obs_keys_to_modality)[k]
                        obs_traj[k] = arr

                    stats = {}
                    for k, v in obs_traj.items():
                        stats[k] = {
                            'n': v.shape[0],
                            'mean': v.mean(axis=0, keepdims=True),
                            'sqdiff': ((v - v.mean(axis=0, keepdims=True)) ** 2).sum(axis=0, keepdims=True),
                        }

                    if merged is None:
                        merged = stats
                    else:
                        for k in merged:
                            n_a = merged[k]['n']
                            n_b = stats[k]['n']
                            n = n_a + n_b
                            delta = stats[k]['mean'] - merged[k]['mean']
                            merged[k]['mean'] = (n_a * merged[k]['mean'] + n_b * stats[k]['mean']) / n
                            merged[k]['sqdiff'] = merged[k]['sqdiff'] + stats[k]['sqdiff'] + delta**2 * n_a * n_b / n
                            merged[k]['n'] = n

        self.normalization_stats = {}
        if merged:
            for k in merged:
                self.normalization_stats[k] = {
                    'mean': merged[k]['mean'].astype(np.float32),
                    'std': (np.sqrt(merged[k]['sqdiff'] / merged[k]['n']) + 1e-6).astype(np.float32),
                }

    def get_normalization_stats(self):
        return dict(self.normalization_stats)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, demo_key, t, label = self.index[idx]
        timesteps = np.arange(t, t + self.window_size)

        cache = self.cache[file_idx][demo_key]
        actions = cache['actions'][timesteps]

        obs = {}
        for k in self.obs_keys:
            cache_key = f'obs/{k}'
            if cache_key in cache:
                obs[k] = cache[cache_key][timesteps]
            else:
                hdf5_data = self.hdf5_files[file_idx][f'data/{demo_key}/obs/{k}']
                obs[k] = hdf5_data[timesteps[0]:timesteps[-1]+1].astype(np.float32)

        return {
            'obs': obs,
            'actions': actions,
            'label': np.float32(label),
        }

    def get_trajectory_data(self, file_idx, demo_key):
        """
        Return the full trajectory for visualization: all windows from one demo.
        Returns obs, actions, label for every valid window start timestep.
        """
        cache = self.cache[file_idx][demo_key]
        demo_len = cache['actions'].shape[0]
        num_windows = max(0, demo_len - self.window_size + 1)
        label = 1.0 if file_idx < self.num_pos else 0.0

        all_obs = {k: [] for k in self.obs_keys}
        all_actions = []

        for t in range(num_windows):
            timesteps = np.arange(t, t + self.window_size)
            all_actions.append(cache['actions'][timesteps])
            for k in self.obs_keys:
                cache_key = f'obs/{k}'
                if cache_key in cache:
                    all_obs[k].append(cache[cache_key][timesteps])
                else:
                    hdf5_data = self.hdf5_files[file_idx][f'data/{demo_key}/obs/{k}']
                    all_obs[k].append(hdf5_data[t:t+self.window_size].astype(np.float32))

        return {
            'obs': {k: np.stack(v) for k, v in all_obs.items()},
            'actions': np.stack(all_actions),
            'label': label,
            'demo_len': demo_len,
        }

    def __del__(self):
        if self._hdf5_files is not None:
            for f in self._hdf5_files:
                f.close()
            self._hdf5_files = None
