# Dataloading
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from seglearn.transform import PadTrunc


class ExerciseDataset(Dataset):
    """Shoulder exercise dataset"""
    
    def __init__(self, npy_file, length=None, transform=None, sanity_check=None):
        self.dataset = np.load(npy_file, allow_pickle=True).item()
        self.seq_length = length
        self.data = self.process_dataset(length)
        self.original_data = self.data.copy()
        self.data = self.data.astype(np.float)
        self.targets = self.dataset['exnum']
        self.targets = self.targets.reshape((self.targets.shape[0],))
        self.original_targets = self.targets.copy()
        self.ex_labels = {0: 'PEN', 1: 'FLEX', 2: 'SCAP', 3: 'ABD', 4: 'IR', 5: 'ER', 6: 'DIAG', 7: 'ROW', 8: 'SLR'}
#         self.subject = self.dataset['subject']
#         self.original_subject = self.subject.copy()
        
        self.transform = transform
        
        if sanity_check is not None:
            self.data = [self.data[sanity_check]]
            self.targets = [self.targets[sanity_check]]

        assert (len(self.data) == len(self.targets))
        
    def process_dataset(self, length):
        shape = [data.shape[0] for data in self.dataset['X']]
        if length is None:
            average_len = round(sum(shape) / len(shape))
            self.seq_length = average_len
        processed, _, _ = PadTrunc(width=self.seq_length).transform(X=self.dataset['X'])
        return processed
    
    def fold(self, fold_indices):
        # Create fold for K-fold validation
        self.data = self.original_data[fold_indices]
        self.targets = self.original_targets[fold_indices]
#         self.subject = self.original_subject[fold_indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        return torch.from_numpy(self.data[idx][:, 1:]), self.targets[idx]