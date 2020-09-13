"""Base class for datasets."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = 'cpu'

FIT_DEFAULT = .8  # Default train split ratio
SEED = 42  # Default seed for splitting

BASEPATH = os.path.join(
    os.path.dirname(__file__),
    os.path.join('..', '..', 'data', 'processed')
)

if not os.path.exists(BASEPATH):
    os.makedirs(BASEPATH)


class NumpyDataset(Dataset):
    """Wrapper for x ndarray with no target."""

    def __init__(self, x):
        self.data = torch.from_numpy(x).float()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def numpy(self, idx=None):
        n = len(self)

        data = self.data.numpy().reshape((n, -1))

        if idx is None:
            return data
        else:
            return data[idx]


class BaseDataset(Dataset):
    """All datasets should subclass BaseDataset."""

    def __init__(self, x, y, split, split_ratio, seed, labels=None):
        if split not in ('train', 'test', 'none'):
            raise Exception('split argument should be "train", "test" or "none"')

        # Get train or test split
        x, y = self.get_split(x, y, split, split_ratio, seed, labels)

        self.data = x.float()
        self.targets = y.float()
        self.latents = None

    def __getitem__(self, index):
        return self.data[index], self.targets[index], index

    def __len__(self):
        return len(self.data)

    def numpy(self, idx=None):
        n = len(self)

        data = self.data.numpy().reshape((n, -1))

        if idx is None:
            return data, self.targets
        else:
            return data[idx], self.targets[idx]

    def get_split(self, x, y, split, split_ratio, seed, labels=None):
        if split == 'none':
            return torch.from_numpy(x), torch.from_numpy(y)

        n = x.shape[0]
        train_idx, test_idx = train_test_split(np.arange(n),
                                               train_size=split_ratio,
                                               random_state=seed,
                                               stratify=labels)

        if split == 'train':
            return torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])
        else:
            return torch.from_numpy(x[test_idx]), torch.from_numpy(y[test_idx])

    def get_latents(self):
        # Return ndarray where columns are latent factors
        return self.latents

    def subset(self, n, random_state):
        # Subset self and return torch dataset
        np.random.seed(random_state)
        sample_mask = np.random.choice(len(self), n, replace=False)

        if self.latents is not None:
            latents = self.latents[sample_mask]
        else:
            latents = None
        return TorchDataset(self.data[sample_mask], self.targets[sample_mask], latents)


class TorchDataset(BaseDataset):
    """Dataset class when splitting is not required and x and y are torch tensors."""

    def __init__(self, x, y, latents):
        self.data = x.float()
        self.targets = y.float()
        self.latents = latents


