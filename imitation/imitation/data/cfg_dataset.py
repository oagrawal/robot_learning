import torch
import torch.utils.data
import numpy as np
from imitation.data.dataset import SequenceDataset

class CFGSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, num_pos=1, **kwargs):
        super().__init__()
        
        # Split the data_paths based on num_pos to identify which is positive and negative
        positive_data_paths = data_paths[:num_pos]
        negative_data_paths = data_paths[num_pos:]
        
        self.pos_dataset = SequenceDataset(data_paths=positive_data_paths, **kwargs)
        self.neg_dataset = SequenceDataset(data_paths=negative_data_paths, **kwargs)
        
        self.pos_len = len(self.pos_dataset)
        self.neg_len = len(self.neg_dataset)
        self.total_len = self.pos_len + self.neg_len

    def get_normalization_stats(self):
        # We exclusively return the normalization stats from the positive dataset.
        # This is because the positive dataset exhibits the desired operational bounds.
        return self.pos_dataset.get_normalization_stats()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if index < self.pos_len:
            # Fetch from positive and assign dummy c=1.0 label
            traj = self.pos_dataset[index]
            traj['c'] = np.array([1.0], dtype=np.float32)
        else:
            # Fetch from negative and assign dummy c=0.0 label
            traj = self.neg_dataset[index - self.pos_len]
            traj['c'] = np.array([0.0], dtype=np.float32)
        
        return traj
