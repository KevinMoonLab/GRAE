"""Synthetic manifold datasets."""
import math
import copy

import torch
import scipy
import numpy as np
from sklearn import datasets

from src.data.base_dataset import BaseDataset, SEED, FIT_DEFAULT

# Default number of samples for synthetic manifolds
SAMPLE = 10000


def slice_3D(x, y, idx, p=1):
    """Utility function to remove a slice from the manifold.

    Args:
        x(ndarray): Input variables.
        y(ndarray): Labels.
        idx(ndarray): Indices of the points to be sliced
        p(float, optional): Probability that a point in idx will be removed.

    Returns:
        (tuple): tuple containing :
                x_1(ndarray): Points not in idx.
                y_1(ndarray): Labels not in idx.
                x_2(ndarray): Points in idx.
                y_2(ndarray): Labels in idx.
    """
    sli = np.zeros(shape=x.shape[0])
    sli[idx] = 1

    sampler = np.random.choice(a=[False, True], size=(sli.shape[0],), p=[1 - p, p])

    sli = np.logical_and(sli, sampler)

    rest = np.logical_not(sli)

    x_1, y_1 = x[rest], y[rest]
    x_2, y_2 = x[sli], y[sli]

    return x_1, y_1, x_2, y_2


class SCurve(BaseDataset):
    """Standard SCurve dataset."""

    def __init__(self, n_samples=SAMPLE, split='none', split_ratio=FIT_DEFAULT,
                 random_state=SEED):
        """Init.

        Args:
            n_samples(int, optional): Number of points to sample from the manifold.
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
        """
        x, y = datasets.make_s_curve(n_samples=n_samples, random_state=random_state)

        super().__init__(x, y, split, split_ratio, random_state)


class Roll(BaseDataset):
    """Standard Swiss Roll dataset.

    Stretched, rotated and rescaled to ensure the manifold is not aligned with the original axes and the data has
    unit variance.
    """

    def __init__(self, n_samples=SAMPLE, split='none', split_ratio=FIT_DEFAULT,
                 random_state=SEED, factor=6, sli_points=250):
        """Init.

        Args:
            n_samples(int, optional): Number of points to sample from the manifold.
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            factor(int, optional): Stretch factor for the roll.
            sli_points(int, optional): Remove sli_points closest to origin on the "length" dimension and use them as the
            test split. Note: Not used by this class, see SwissRoll. Roll uses uniform sampling to determine the splits.
        """
        x, y = datasets.make_swiss_roll(n_samples=n_samples, random_state=random_state)

        # Backup first axis, as it represents one of the underlying latent
        # variable we aim to recover
        y_pure = copy.deepcopy(x[:, 1])
        latents = np.vstack((y, y_pure)).T

        # Normalize
        x = scipy.stats.zscore(x)

        # Get absolute distance from origin
        ab = np.abs(x[:, 1])
        sort = np.argsort(ab)

        # Take the sli_points points closest to origin
        # This is not used by the base class, but will be used by the SwissRoll
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

        super().__init__(x, latents, split, split_ratio, random_state)

        # Latent variables are the coordinate on the "length" dimension and the color given by Sklearn
        # Both parametrize the intrinsic plane
        self.latents = self.targets.numpy()

        # Only keep one latent as target for compatibility with other datasets
        self.targets = self.targets[:, 0]


class SwissRoll(Roll):
    """Swiss Roll class where the test split is a thin 'ribbon' of sli_points
    points removed from the middle of the manifold to test out of distribution robustness.

    This is the dataset used in the GRAE paper."""

    def __init__(self, n_samples=SAMPLE, sli_points=250, split='none',
                 split_ratio=FIT_DEFAULT, random_state=SEED):
        """Init.

        Args:
            n_samples(int, optional): Number of points to sample from the manifold.
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            sli_points(int, optional): Remove sli_points closest to origin on the "length" dimension and use them as
            the test split.
        """

        super().__init__(n_samples, split, split_ratio=split_ratio, random_state=random_state, sli_points=sli_points)

    def get_split(self, x, y, split, split_ratio, random_state, labels=None):
        """Split dataset.

        Args:
            x(ndarray): Input features.
            y(ndarray): Targets.
            split(str): Name of split.
            split_ratio(float): Ratio to use for train split. Test split ratio is 1 - split_ratio.
            random_state(int): To set random_state values for reproducibility.
            labels(ndarray, optional): Ignored.

        Returns:
            (tuple): tuple containing :
                    x(ndarray): Input variables in requested split.
                    y(ndarray): Target variable in requested split.
        """
        if split == 'none':
            return torch.from_numpy(x), torch.from_numpy(y)

        x_train, y_train, x_test, y_test = slice_3D(x, y, self.test_idx)

        if split == 'train':
            return torch.from_numpy(x_train), torch.from_numpy(y_train)
        else:
            return torch.from_numpy(x_test), torch.from_numpy(y_test)


"""Following is from the Topological Autoencoders paper from Moor & al to unit test our TopoAE class.

Copied from their source code. Available here : https://osf.io/abuce/?view_only=f16d65d3f73e4918ad07cdd08a1a0d4b"""


def dsphere(n=100, d=2, r=1, noise=None):
    """
    Sample `n` data points on a d-sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in
        this high dimensional space.
    """
    data = np.random.randn(n, d + 1)

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data ** 2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    return data


def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, seed=42):
    np.random.seed(seed)

    # it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance = 10 / np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d + 1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres - 1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 10 * n_samples  # int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r * 5)
    spheres.append(big)
    n_datapoints += n_samples_big

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    return dataset, labels


class Spheres(BaseDataset):
    """Small high dimensional spheres in a big sphere, as presented in the Topological Autoencoders paper."""

    def __init__(self, split='none', split_ratio=FIT_DEFAULT,
                 random_state=SEED):
        """Init.

        Args:
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
        """
        x, y = create_sphere_dataset(seed=random_state)

        super().__init__(x, y, split, split_ratio, random_state)
