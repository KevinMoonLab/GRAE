"""Synthetic manifold datasets."""
import math

import torch
import scipy
import copy
import numpy as np
from sklearn import datasets

from src.data.base import BaseDataset, SEED, FIT_DEFAULT

SAMPLE = 10000


def slice_3D(X, Y, idx, p=1):
    """Utility function to slice manifolds."""
    sli = np.zeros(shape=X.shape[0])
    sli[idx] = 1

    sampler = np.random.choice(a=[False, True], size=(sli.shape[0],), p=[1 - p, p])

    sli = np.logical_and(sli, sampler)

    rest = np.logical_not(sli)

    x_2, y_2 = X[rest], Y[rest]
    x_3, y_3 = X[sli], Y[sli]

    return x_2, y_2, x_3, y_3


class SCurve(BaseDataset):
    """Standard SCurve dataset."""

    def __init__(self, n_samples=SAMPLE, split='none', split_ratio=FIT_DEFAULT,
                 seed=SEED):
        x, y = datasets.make_s_curve(n_samples=n_samples, random_state=seed)

        super().__init__(x, y, split, split_ratio, seed)


class Roll(BaseDataset):
    """Standard Swiss Roll dataset."""

    def __init__(self, n_samples=SAMPLE, split='none', split_ratio=FIT_DEFAULT,
                 seed=SEED, factor=6, sli_points=250):
        x, y = datasets.make_swiss_roll(n_samples=n_samples, random_state=seed)

        # Backup first axis, as it represents one of the underlying latent
        # variable we aim to recover
        self.y_pure = copy.deepcopy(x[:, 1])

        # Normalize
        x = scipy.stats.zscore(x)

        # Get absolute distance from origin
        ab = np.abs(x[:, 1])
        sort = np.argsort(ab)

        # Take the sli_points points closest to origin
        # This is not used by the base class, but will be used by the Ribbons
        # children class to remove a thin slice from the roll
        self.test_idx = sort[0:sli_points]

        # Apply rotation  to achieve same variance on all axes
        x[:, 1] *= factor

        theta = math.pi / 4
        phi = math.pi / 3.3
        rho = 0

        cos = math.cos(theta)
        sin = math.sin(theta)

        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        cos_rho = math.cos(rho)
        sin_rho = math.sin(rho)

        rot = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])

        rot_2 = np.array([[cos_phi, -sin_phi, 0], [sin_phi, cos_phi, 0], [0, 0, 1]])
        rot_3 = np.array([[1, 0, 0], [0, cos_rho, -sin_rho], [0, sin_rho, cos_rho]])

        x = x @ rot_2 @ rot_3 @ rot
        x = scipy.stats.zscore(x)  # Normalize for true unit variance

        super().__init__(x, y, split, split_ratio, seed)

    def get_latents(self):
        # First source is coloring, second is the axis 1 (length of the roll)
        return np.vstack((self.targets.numpy().flatten(), self.y_pure)).T


class SwissRoll(Roll):
    """Swiss Roll class where the test split is a thin 'ribbon' of sli_points
    points removed from the middle of the manifold.

    This is the dataset used in the GRAE paper."""

    def __init__(self, n_samples=SAMPLE, sli_points=250, split='none',
                 split_ratio=FIT_DEFAULT, seed=SEED):

        super().__init__(n_samples, split, split_ratio=split_ratio, seed=seed,
                         factor=6, sli_points=sli_points)

    def get_split(self, x, y, split, split_ratio, seed):
        if split == 'none':
            return torch.from_numpy(x), torch.from_numpy(y)

        x_train, y_train, x_test, y_test = slice_3D(x, y, self.test_idx)
        train_mask = np.full(x.shape[0], fill_value=True)
        train_mask[self.test_idx] = False
        test_mask = np.logical_not(train_mask)

        if split == 'train':
            self.y_pure = self.y_pure[train_mask]
            return torch.from_numpy(x_train), torch.from_numpy(y_train)
        else:
            self.y_pure = self.y_pure[test_mask]
            return torch.from_numpy(x_test), torch.from_numpy(y_test)

