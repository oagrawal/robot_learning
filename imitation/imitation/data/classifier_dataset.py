import json
import os
import h5py
import numpy as np
import torch.utils.data
from tqdm import tqdm

from imitation.utils.general_utils import AttrDict
from imitation.utils.obs_utils import process_obs_dict


class ClassifierDataset(torch.utils.data.Dataset):
    """
    Trajectory-level success classifier samples aligned with FM inference:
      obs: last n_obs_steps frames (low_dim + optional RGB)
      actions: action_chunk_size actions from anchor t (clamped at demo end)

    Labels: 1.0 success demo, 0.0 failure demo. Splits are by whole demos only.

    Optional 3-way split: set n_val_demos_per_class and n_test_demos_per_class (e.g. 30, 30).
    Optional hard val/test: hard_val_json / hard_test_json (paths) with failure_regions.
      Negative val/test rows are only timesteps inside those JSON intervals (not a random
      % of failure demos). Each (file_idx, demo_key) in the JSON must exist as a failure
      trajectory in data_paths. Positive rows are random anchors from the val/test success
      split only, matched 1:1 to negatives.

    Leakage: by default (exclude_hard_json_failures_from_train=True), any failure
    (file_idx, demo_key) that appears in hard_val_json or hard_test_json is removed
    from the train failure demo list so those trajectories are not trained on.
    """

    SPLIT = AttrDict(train=0.80, val=0.20)

    def __init__(
        self,
        data_paths,
        num_pos=1,
        n_obs_steps=2,
        action_chunk_size=8,
        max_demo_len=None,
        obs_keys_to_modality=None,
        obs_keys_to_normalize=None,
        split='train',
        n_val_demos_per_class=None,
        n_test_demos_per_class=None,
        hard_val_json=None,
        hard_test_json=None,
        ablation_mode='action_and_state',
        hard_eval_seed=42,
        **kwargs,
    ):
        super().__init__()

        if kwargs.pop('window_size', None) is not None:
            import warnings
            warnings.warn(
                "ClassifierDataset: window_size is ignored; use n_obs_steps and action_chunk_size",
                DeprecationWarning,
                stacklevel=2,
            )

        obs_keys_to_modality = obs_keys_to_modality or {}
        obs_keys_to_normalize = obs_keys_to_normalize or {}

        assert isinstance(data_paths, list)
        self.hdf5_paths = [os.path.expanduser(p) for p in data_paths]
        self.obs_keys = tuple(obs_keys_to_modality.keys())
        self.obs_keys_to_modality = obs_keys_to_modality
        self.n_obs_steps = int(n_obs_steps)
        self.action_chunk_size = int(action_chunk_size)
        self.max_demo_len = max_demo_len
        self.split = split
        self.num_pos = num_pos
        self.ablation_mode = ablation_mode
        self.hard_eval_seed = int(hard_eval_seed)

        self.n_val_demos_per_class = n_val_demos_per_class
        self.n_test_demos_per_class = n_test_demos_per_class
        self.hard_val_json = os.path.expanduser(hard_val_json) if hard_val_json else None
        self.hard_test_json = os.path.expanduser(hard_test_json) if hard_test_json else None

        kwargs.pop("hard_json_neg_demo_pool", None)
        self.exclude_hard_json_failures_from_train = bool(
            kwargs.pop("exclude_hard_json_failures_from_train", True)
        )

        self._build_index()
        self._cache_low_dim()
        self._compute_normalization_stats(list(obs_keys_to_normalize.keys()))

    def _three_way_split(self):
        return (
            self.n_val_demos_per_class is not None
            and self.n_test_demos_per_class is not None
        )

    def _load_hard_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('failure_regions', data)

    def _hard_json_excluded_failure_demos(self):
        """(file_idx, demo_key) union from existing hard val/test JSON files."""
        out = set()
        for path in (self.hard_val_json, self.hard_test_json):
            if path and os.path.isfile(path):
                for r in self._load_hard_json(path):
                    out.add((int(r["file_idx"]), r["demo_key"]))
        return out

    def _filter_neg_train(self, neg_train, excluded):
        if not excluded or not self.exclude_hard_json_failures_from_train:
            return neg_train
        out = [d for d in neg_train if (d[0], d[1]) not in excluded]
        n_drop = len(neg_train) - len(out)
        if n_drop:
            print(
                f"ClassifierDataset [{self.split}]: excluding {n_drop} failure demo(s) "
                f"listed in hard val/test JSON from training negatives"
            )
        if not out:
            raise ValueError(
                "All failure demos in the train split are listed in hard_val_json / "
                "hard_test_json; nothing left for training negatives. "
                "Lower the val/test fraction, add more failure data, or set "
                "exclude_hard_json_failures_from_train=False."
            )
        return out

    def _build_index_hard(self, json_path, pos_pool, neg_demos_all):
        """
        pos_pool: val/test success demos (file_idx, demo_key, demo_len).

        neg_demos_all: every failure trajectory in data_paths (used only to verify JSON
        demo keys exist). Negative index rows are built solely from JSON [start_t, end_t].
        """
        neg_demo_set = {(fi, dk) for fi, dk, _ in neg_demos_all}
        if not pos_pool:
            raise ValueError("Hard eval requires at least one success demo in this split")

        regions = self._load_hard_json(json_path)
        neg_entries = []
        for r in regions:
            file_idx = int(r['file_idx'])
            demo_key = r['demo_key']
            start_t = int(r['start_t'])
            end_t = int(r['end_t'])
            if (file_idx, demo_key) not in neg_demo_set:
                raise ValueError(
                    f"Annotated failure ({file_idx}, {demo_key}) not found among failure trajectories "
                    f"in data_paths"
                )
            for t in range(start_t, end_t + 1):
                neg_entries.append((file_idx, demo_key, t, 0.0))

        n_neg = len(neg_entries)
        if n_neg == 0:
            raise ValueError(f"No negative rows from {json_path}")

        rng = np.random.RandomState(self.hard_eval_seed)
        pos_entries = []
        for _ in range(n_neg):
            fi, dk, dlen = pos_pool[rng.randint(0, len(pos_pool))]
            t = rng.randint(0, max(1, dlen))
            pos_entries.append((fi, dk, t, 1.0))

        self.index = neg_entries + pos_entries
        self.sample_weights = np.ones(len(self.index), dtype=np.float64)
        print(
            f"ClassifierDataset [{self.split}] HARD EVAL: {n_neg} neg + {n_neg} pos = {len(self.index)} "
            f"(from {json_path})"
        )

    def _build_index_standard(self, selected):
        self.index = []
        for (file_idx, demo_key, demo_len), label in selected:
            effective_len = demo_len
            if self.max_demo_len is not None and label == 0.0:
                effective_len = min(demo_len, self.max_demo_len)
            for t in range(effective_len):
                self.index.append((file_idx, demo_key, t, label))
        self._build_sample_weights()

    def _build_index(self):
        self.index = []
        self.hdf5_use_swmr = True

        pos_demos = []
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

        use_three = self._three_way_split()
        if use_three:
            n_val = int(self.n_val_demos_per_class)
            n_test = int(self.n_test_demos_per_class)

            def split_three(demos):
                n = len(demos)
                if n < n_val + n_test + 1:
                    raise ValueError(
                        f"Need at least n_val+n_test+1={n_val + n_test + 1} demos, got {n}"
                    )
                n_train = n - n_val - n_test
                return demos[:n_train], demos[n_train:n_train + n_val], demos[n_train + n_val:]

            pos_train, pos_val, pos_test = split_three(pos_demos)
            neg_train, neg_val, neg_test = split_three(neg_demos)

            if self.split == 'train':
                excluded = self._hard_json_excluded_failure_demos()
                neg_train_f = self._filter_neg_train(neg_train, excluded)
                selected = [(d, 1.0) for d in pos_train] + [(d, 0.0) for d in neg_train_f]
            elif self.split == 'val':
                if self.hard_val_json and os.path.isfile(self.hard_val_json):
                    self._build_index_hard(self.hard_val_json, pos_val, neg_demos)
                    return
                selected = [(d, 1.0) for d in pos_val] + [(d, 0.0) for d in neg_val]
            elif self.split == 'test':
                if self.hard_test_json and os.path.isfile(self.hard_test_json):
                    self._build_index_hard(self.hard_test_json, pos_test, neg_demos)
                    return
                selected = [(d, 1.0) for d in pos_test] + [(d, 0.0) for d in neg_test]
            else:
                raise ValueError(f"Unknown split {self.split}")
        else:
            if self.split == 'test':
                raise ValueError(
                    "split='test' requires n_val_demos_per_class and n_test_demos_per_class"
                )

            def split_two(demos):
                n_train = max(1, int(self.SPLIT.train * len(demos)))
                return demos[:n_train], demos[n_train:]

            pos_train, pos_val = split_two(pos_demos)
            neg_train, neg_val = split_two(neg_demos)

            if self.split == 'train':
                excluded = self._hard_json_excluded_failure_demos()
                neg_train_f = self._filter_neg_train(neg_train, excluded)
                selected = [(d, 1.0) for d in pos_train] + [(d, 0.0) for d in neg_train_f]
            else:
                if self.hard_val_json and os.path.isfile(self.hard_val_json):
                    self._build_index_hard(self.hard_val_json, pos_val, neg_demos)
                    return
                selected = [(d, 1.0) for d in pos_val] + [(d, 0.0) for d in neg_val]

        self._build_index_standard(selected)

    def _build_sample_weights(self):
        labels = np.array([entry[3] for entry in self.index])
        n_pos = (labels == 1.0).sum()
        n_neg = (labels == 0.0).sum()
        total = n_pos + n_neg

        w_pos = total / (2.0 * max(n_pos, 1))
        w_neg = total / (2.0 * max(n_neg, 1))

        self.sample_weights = np.where(labels == 1.0, w_pos, w_neg)
        print(
            f"ClassifierDataset [{self.split}]: {n_pos} pos windows, {n_neg} neg windows "
            f"(weights: pos={w_pos:.2f}, neg={w_neg:.2f})"
        )

    def _cache_low_dim(self):
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
                            self.cache[file_idx][demo_key][f'obs/{k}'] = (
                                demo[f'obs/{k}'][:].astype(np.float32)
                            )

        self._hdf5_files = None

    @property
    def hdf5_files(self):
        if self._hdf5_files is None:
            self._hdf5_files = [
                h5py.File(p, 'r', swmr=True, libver='latest') for p in self.hdf5_paths
            ]
        return self._hdf5_files

    def _compute_normalization_stats(self, obs_keys_to_normalize):
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
                            'sqdiff': ((v - v.mean(axis=0, keepdims=True)) ** 2).sum(
                                axis=0, keepdims=True
                            ),
                        }

                    if merged is None:
                        merged = stats
                    else:
                        for k in merged:
                            n_a = merged[k]['n']
                            n_b = stats[k]['n']
                            n = n_a + n_b
                            delta = stats[k]['mean'] - merged[k]['mean']
                            merged[k]['mean'] = (
                                n_a * merged[k]['mean'] + n_b * stats[k]['mean']
                            ) / n
                            merged[k]['sqdiff'] = (
                                merged[k]['sqdiff']
                                + stats[k]['sqdiff']
                                + delta**2 * n_a * n_b / n
                            )
                            merged[k]['n'] = n

        self.normalization_stats = {}
        if merged:
            for k in merged:
                self.normalization_stats[k] = {
                    'mean': merged[k]['mean'].astype(np.float32),
                    'std': (np.sqrt(merged[k]['sqdiff'] / merged[k]['n']) + 1e-6).astype(
                        np.float32
                    ),
                }

    def get_normalization_stats(self):
        return dict(self.normalization_stats)

    def __len__(self):
        return len(self.index)

    def _normalize_actions(self, actions):
        if 'actions' in self.normalization_stats:
            mean = self.normalization_stats['actions']['mean']
            std = self.normalization_stats['actions']['std']
            return (actions - mean) / std
        return actions

    def _read_obs_frame(self, file_idx, demo_key, t):
        obs = {}
        cache = self.cache[file_idx][demo_key]
        for k in self.obs_keys:
            cache_key = f'obs/{k}'
            if cache_key in cache:
                obs[k] = cache[cache_key][t].copy()
            else:
                hdf5_data = self.hdf5_files[file_idx][f'data/{demo_key}/obs/{k}']
                obs[k] = hdf5_data[t].astype(np.float32)
        return obs

    def _stack_obs_window(self, file_idx, demo_key, anchor_t):
        cache = self.cache[file_idx][demo_key]
        demo_len = cache['actions'].shape[0]
        obs_window = {k: [] for k in self.obs_keys}
        for i in range(self.n_obs_steps):
            tt = anchor_t - self.n_obs_steps + 1 + i
            tt = max(0, min(tt, demo_len - 1))
            frame = self._read_obs_frame(file_idx, demo_key, tt)
            for k in self.obs_keys:
                obs_window[k].append(frame[k])
        return {k: np.stack(obs_window[k], axis=0) for k in self.obs_keys}

    def _action_chunk(self, file_idx, demo_key, anchor_t):
        cache = self.cache[file_idx][demo_key]
        demo_len = cache['actions'].shape[0]
        idx = np.arange(anchor_t, anchor_t + self.action_chunk_size)
        idx = np.minimum(idx, demo_len - 1)
        raw = cache['actions'][idx]
        return self._normalize_actions(raw.astype(np.float32))

    def __getitem__(self, idx):
        file_idx, demo_key, t, label = self.index[idx]
        t = int(t)

        obs = self._stack_obs_window(file_idx, demo_key, t)
        actions = self._action_chunk(file_idx, demo_key, t)

        if self.ablation_mode == 'action_only':
            for k in obs:
                obs[k] = np.zeros_like(obs[k])
        elif self.ablation_mode == 'state_only':
            actions = np.zeros_like(actions)
        elif self.ablation_mode == 'shuffled_actions':
            rng = np.random.RandomState(
                (hash((file_idx, demo_key, t, self.split)) % (2**31))
            )
            perm = rng.permutation(self.action_chunk_size)
            actions = actions[perm].copy()
        elif self.ablation_mode != 'action_and_state':
            raise ValueError(f"Unknown ablation_mode {self.ablation_mode}")

        return {
            'obs': obs,
            'actions': actions,
            'label': np.float32(label),
        }

    def get_trajectory_data(self, file_idx, demo_key):
        cache = self.cache[file_idx][demo_key]
        demo_len = cache['actions'].shape[0]
        label = 1.0 if file_idx < self.num_pos else 0.0

        all_obs = {k: [] for k in self.obs_keys}
        all_actions = []

        for t in range(demo_len):
            ow = self._stack_obs_window(file_idx, demo_key, t)
            ac = self._action_chunk(file_idx, demo_key, t)
            for k in self.obs_keys:
                all_obs[k].append(ow[k])
            all_actions.append(ac)

        return {
            'obs': {k: np.stack(all_obs[k], axis=0) for k in self.obs_keys},
            'actions': np.stack(all_actions, axis=0),
            'label': label,
            'demo_len': demo_len,
        }

    def __del__(self):
        if getattr(self, '_hdf5_files', None) is not None:
            for f in self._hdf5_files:
                f.close()
            self._hdf5_files = None
