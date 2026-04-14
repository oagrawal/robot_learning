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
    SPLIT = AttrDict(train=0.80, val=0.20)

    def __init__(self, data_paths, num_pos=1, window_size=3,
                 max_demo_len=None,
                 obs_keys_to_modality={}, obs_keys_to_normalize={},
                 split='train', **kwargs):
        super().__init__()

        assert isinstance(data_paths, list)
        self.hdf5_paths = [os.path.expanduser(p) for p in data_paths]
        self.obs_keys = tuple(obs_keys_to_modality.keys())
        self.obs_keys_to_modality = obs_keys_to_modality
        self.window_size = window_size
        self.max_demo_len = max_demo_len
        self.split = split
        self.num_pos = num_pos

        self._build_index()
        self._cache_low_dim()
        self._compute_normalization_stats(list(obs_keys_to_normalize.keys()))

    def _build_index(self):
        """Build a flat index: each entry is (file_idx, demo_key, timestep, label).
        Split is done at the demo level with stratification so both classes
        appear in both train and val."""
        self.index = []
        self.hdf5_use_swmr = True

        pos_demos = []  # (file_idx, demo_key, demo_len)
        neg_demos = []

        for file_idx, path in enumerate(self.hdf5_paths):
            is_pos = file_idx < self.num_pos
            with h5py.File(path, 'r', swmr=True, libver='latest') as f:
                demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
                for demo_key in demos:
                    demo_len = f[f'data/{demo_key}/actions'].shape[0]
                    entry = (file_idx, demo_key, demo_len)
                    if is_pos:
                        pos_demos.append(entry)
                    else:
                        neg_demos.append(entry)

        rng = np.random.RandomState(42)
        rng.shuffle(pos_demos)
        rng.shuffle(neg_demos)

        def split_demos(demos):
            n_train = max(1, int(self.SPLIT.train * len(demos)))
            return demos[:n_train], demos[n_train:]

        pos_train, pos_val = split_demos(pos_demos)
        neg_train, neg_val = split_demos(neg_demos)

        if self.split == 'train':
            selected = [(d, 1.0) for d in pos_train] + [(d, 0.0) for d in neg_train]
        else:
            selected = [(d, 1.0) for d in pos_val] + [(d, 0.0) for d in neg_val]

        for (file_idx, demo_key, demo_len), label in selected:
            effective_len = demo_len
            if self.max_demo_len is not None and label == 0.0:
                effective_len = min(demo_len, self.max_demo_len)
            num_windows = max(0, effective_len - self.window_size + 1)
            for t in range(num_windows):
                self.index.append((file_idx, demo_key, t, label))

        self._build_sample_weights()

    def _build_sample_weights(self):
        """Compute per-sample weights so that each class contributes equally
        to the training loss despite different window counts."""
        labels = np.array([entry[3] for entry in self.index])
        n_pos = (labels == 1.0).sum()
        n_neg = (labels == 0.0).sum()
        total = n_pos + n_neg

        w_pos = total / (2.0 * max(n_pos, 1))
        w_neg = total / (2.0 * max(n_neg, 1))

        self.sample_weights = np.where(labels == 1.0, w_pos, w_neg)
        print(f"ClassifierDataset [{self.split}]: {n_pos} pos windows, {n_neg} neg windows "
              f"(weights: pos={w_pos:.2f}, neg={w_neg:.2f})")

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
        """Compute normalization stats from positive (success) data only.
        Also computes action stats so the classifier can operate in
        the same normalized action space as the flow policy."""
        merged = None
        for file_idx in range(self.num_pos):
            path = self.hdf5_paths[file_idx]
            with h5py.File(path, 'r', swmr=True, libver='latest') as f:
                demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
                for demo_key in demos:
                    data_dict = {}
                    for k in obs_keys_to_normalize:
                        arr = f[f'data/{demo_key}/obs/{k}'][()].astype(np.float32)
                        arr = process_obs_dict({k: arr}, self.obs_keys_to_modality)[k]
                        data_dict[k] = arr

                    data_dict['actions'] = f[f'data/{demo_key}/actions'][()].astype(np.float32)

                    stats = {}
                    for k, v in data_dict.items():
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

    def _normalize_actions(self, actions):
        """Normalize actions using precomputed stats (same as flow policy)."""
        if 'actions' in self.normalization_stats:
            mean = self.normalization_stats['actions']['mean']
            std = self.normalization_stats['actions']['std']
            return (actions - mean) / std
        return actions

    def __getitem__(self, idx):
        file_idx, demo_key, t, label = self.index[idx]
        timesteps = np.arange(t, t + self.window_size)

        cache = self.cache[file_idx][demo_key]
        actions = self._normalize_actions(cache['actions'][timesteps])

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
            all_actions.append(self._normalize_actions(cache['actions'][timesteps]))
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
