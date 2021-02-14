"""Base class for datasets.

All data will be saved to a data/processed/dataset_name folder."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FIT_DEFAULT = .85  # Default train/test split ratio
SEED = 42  # Default seed for splitting

DEFAULT_PATH = os.path.join(os.getcwd(), 'data')


class FromNumpyDataset(Dataset):
    """Torch Dataset Wrapper for x ndarray with no target."""

    def __init__(self, x):
        """Create torch wraper dataset form simple ndarray.

        Args:
            x (ndarray): Input variables.
        """
        self._data = torch.from_numpy(x).float()

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def numpy(self, idx=None):
        """Get dataset as ndarray.

        Specify indices to return a subset of the dataset, otherwise return whole dataset.

        Args:
            idx(int, optional): Specify index or indices to return.

        Returns:
            ndarray: Return flattened dataset as a ndarray.

        """
        n = len(self)

        data = self._data.numpy().reshape((n, -1))

        if idx is None:
            return data
        else:
            return data[idx]


class BaseDataset(Dataset):
    """Template class for all datasets in the project.

    All datasets should subclass BaseDataset, which contains built-in splitting utilities."""

    def __init__(self, x, y, split, split_ratio, random_state, labels=None):
        """Init.

        Set the split parameter to 'train' or 'test' for the object to hold the desired split. split='none' will keep
        the entire dataset in the attributes.

        Args:
            x(ndarray): Input features.
            y(ndarray): Targets.
            split(str): Name of split.
            split_ratio(float): Ratio to use for train split. Test split ratio is 1 - split_ratio.
            random_state(int): To set random_state values for reproducibility.
            labels(ndarray, optional): Specify labels for stratified splits.
        """
        if split not in ('train', 'test', 'none'):
            raise ValueError('split argument should be "train", "test" or "none"')

        # Get train or test split
        x, y = self.get_split(x, y, split, split_ratio, random_state, labels)

        self.data = x.float()
        self.targets = y.float()  # One target variable. Used mainly for coloring.
        self.latents = None  # Arbitrary number of continuous ground truth variables. Used for computing metrics.

        # Arbitrary number of label ground truth variables. Used for computing metrics.
        # Should range from 0 to no_of_classes -1
        self.labels = None
        self.is_radial = []  # Indices of latent variable requiring polar conversion when probing (e.g. Teapot, RotatedDigits)
        self.partition = True  # If labels should be used to partition the data before regressing latent factors. See score.EmbeddingProber.

    def __getitem__(self, index):
        return self.data[index], self.targets[index], index

    def __len__(self):
        return len(self.data)

    def numpy(self, idx=None):
        """Get dataset as ndarray.

        Specify indices to return a subset of the dataset, otherwise return whole dataset.

        Args:
            idx(int, optional): Specify index or indices to return.

        Returns:
            ndarray: Return flattened dataset as a ndarray.

        """
        n = len(self)

        data = self.data.numpy().reshape((n, -1))

        if idx is None:
            return data, self.targets.numpy()
        else:
            return data[idx], self.targets[idx].numpy()

    def get_split(self, x, y, split, split_ratio, random_state, labels=None):
        """Split dataset.

        Args:
            x(ndarray): Input features.
            y(ndarray): Targets.
            split(str): Name of split.
            split_ratio(float): Ratio to use for train split. Test split ratio is 1 - split_ratio.
            random_state(int): To set random_state values for reproducibility.
            labels(ndarray, optional): Specify labels for stratified splits.

        Returns:
            (tuple): tuple containing :
                    x(ndarray): Input variables in requested split.
                    y(ndarray): Target variable in requested split.
        """
        if split == 'none':
            return torch.from_numpy(x), torch.from_numpy(y)

        n = x.shape[0]
        train_idx, test_idx = train_test_split(np.arange(n),
                                               train_size=split_ratio,
                                               random_state=random_state,
                                               stratify=labels)

        if split == 'train':
            return torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])
        else:
            return torch.from_numpy(x[test_idx]), torch.from_numpy(y[test_idx])

    def get_latents(self):
        """Latent variable getter.

        Returns:
            latents(ndarray): Latent variables for each sample.
        """
        return self.latents

    def random_subset(self, n, random_state):
        """Random subset self and return corresponding dataset object.

        Args:
            n(int): Number of samples to subset.
            random_state(int): Seed for reproducibility

        Returns:
            Subset(TorchDataset) : Random subset.

        """

        np.random.seed(random_state)
        sample_mask = np.random.choice(len(self), n, replace=False)

        next_latents = self.latents[sample_mask] if self.latents is not None else None
        next_labels = self.labels[sample_mask] if self.labels is not None else None

        return NoSplitBaseDataset(self.data[sample_mask], self.targets[sample_mask], next_latents, next_labels)

    def validation_split(self, ratio=.15 / FIT_DEFAULT, random_state=42):
        """Randomly subsample validation split in self.

        Return both train split and validation split as two different BaseDataset objects.

        Args:
            ratio(float): Ratio of train split to allocate to validation split. Default option is to sample 15 % of
            full dataset, by adjusting with the initial train/test ratio.
            random_state(int): Seed for sampling.

        Returns:
            (tuple) tuple containing:
                x_train(BaseDataset): Train set.
                x_val(BaseDataset): Val set.

        """

        np.random.seed(random_state)
        sample_mask = np.random.choice(len(self), int(ratio * len(self)), replace=False)
        val_mask = np.full(len(self), False, dtype=bool)
        val_mask[sample_mask] = True
        train_mask = np.logical_not(val_mask)
        next_latents_train = self.latents[train_mask] if self.latents is not None else None
        next_latents_val = self.latents[val_mask] if self.latents is not None else None
        next_labels_train = self.labels[train_mask] if self.labels is not None else None
        next_labels_val = self.labels[val_mask] if self.labels is not None else None

        x_train = NoSplitBaseDataset(self.data[train_mask], self.targets[train_mask],
                                     next_latents_train, next_labels_train)
        x_val = NoSplitBaseDataset(self.data[val_mask], self.targets[val_mask],
                                   next_latents_val, next_labels_val)

        return x_train, x_val


class NoSplitBaseDataset(BaseDataset):
    """BaseDataset class when splitting is not required and x and y are already torch tensors."""

    def __init__(self, x, y, latents, labels):
        """Init.

        Args:
            x(ndarray): Input variables.
            y(ndarray): Target variable. Used for coloring.
            latents(ndarray): Other continuous target variable. Used for metrics.
            labels(ndarray): Other label target variable. Used for metrics.
        """
        self.data = x.float()
        self.targets = y.float()
        self.latents = latents
        self.labels = labels
