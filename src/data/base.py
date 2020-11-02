"""Base class for datasets.

All data will be saved to a data/processed/dataset_name folder."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FIT_DEFAULT = .8  # Default train split ratio
SEED = 42  # Default seed for splitting

BASEPATH = os.path.join(
    os.path.dirname(__file__),
    os.path.join('..', '..', 'data', 'processed')
)

# Create BASEPATH if needed
if not os.path.exists(BASEPATH):
    os.makedirs(BASEPATH)


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
        self.latents = None  # Arbitrary number of ground truth variables. Used for computing metrics.

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
            return data, self.targets
        else:
            return data[idx], self.targets[idx]

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

        if self.latents is not None:
            latents = self.latents[sample_mask]
        else:
            latents = None
        return NoSplitBaseDataset(self.data[sample_mask], self.targets[sample_mask], latents)


class NoSplitBaseDataset(BaseDataset):
    """BaseDataset class when splitting is not required and x and y are already torch tensors."""
    def __init__(self, x, y, latents):
        """Init.

        Args:
            x(ndarray): Input variables.
            y(ndarray): Target variable. Used for coloring.
            latents(ndarray): Other target variable. Used for metrics.
        """
        self.data = x.float()
        self.targets = y.float()
        self.latents = latents
