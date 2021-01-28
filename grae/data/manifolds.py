"""Synthetic manifold datasets."""
import math
import copy
import os
import requests

import torch
import scipy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from sklearn import datasets
import phate

from grae.data.base_dataset import BaseDataset, SEED, FIT_DEFAULT, DEFAULT_PATH

# Default number of samples for synthetic manifolds
SAMPLE = 10000


# Utility functions
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

    sampler = np.random.choice(a=[False, True], size=(sli.shape[0],),
                               p=[1 - p, p])

    sli = np.logical_and(sli, sampler)

    rest = np.logical_not(sli)

    x_1, y_1 = x[rest], y[rest]
    x_2, y_2 = x[sli], y[sli]

    return x_1, y_1, x_2, y_2


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
      
    Credits to Karlo from https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def surface_plot(x, y, s=20, tilt=30, rotation=-80, edgecolor='k'):
    """3D plot of the data

    Args:
        x(ndarray): Points.
        y(ndarray): Labels for coloring.
        s(int): Marker size.
        tilt(int): Inclination towards observer.
        rotation(rotation): Rotation angle.
        edgecolor(str): Edge color.

    """
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(tilt, rotation)
    ax.scatter(*x.T,
               cmap="jet",
               c=y,
               s=s,
               edgecolor=edgecolor)
    set_axes_equal(ax)


class Surface(BaseDataset):
    """Class for 2D surfaces embedded in 3D"""

    def plot(self, s=20, tilt=30, rotation=-80, edgecolor='k'):
        """3D plot of the data

        Args:
            s(int): Marker size.
            tilt(int): Inclination towards observer.
            rotation(rotation): Rotation angle.
            edgecolor(str): Edge color.

        """
        x, y = self.numpy()
        surface_plot(x, y, s, tilt, rotation, edgecolor)
        plt.show()


class SCurve(Surface):
    """Standard SCurve dataset."""

    def __init__(self, n_samples=SAMPLE, split='none', split_ratio=FIT_DEFAULT,
                 random_state=SEED, data_path=DEFAULT_PATH):
        """Init.

        Args:
            n_samples(int, optional): Number of points to sample from the manifold.
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            data_path(str, optional): Unused. Only to share same signature with other datasets.
        """
        x, y = datasets.make_s_curve(n_samples=n_samples,
                                     random_state=random_state)

        super().__init__(x, y, split, split_ratio, random_state)


class FullSwissRoll(Surface):
    """Standard Swiss Roll dataset.

    Stretched, rotated and rescaled to ensure the manifold is not aligned with the original axes and the data has
    unit variance.
    """

    def __init__(self, n_samples=SAMPLE, split='none', split_ratio=FIT_DEFAULT,
                 random_state=SEED, factor=6, sli_points=250,
                 data_path=DEFAULT_PATH):
        """Init.

        Args:
            n_samples(int, optional): Number of points to sample from the manifold.
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            factor(int, optional): Stretch factor for the roll.
            sli_points(int, optional): Remove sli_points closest to origin on the "length" dimension and use them as the
            test split. Note: Not used by this class, see SwissRoll. FullSwissRoll uses uniform sampling to determine the splits.
            data_path(str, optional): Unused. Only to share same signature with other datasets.
        """
        x, y = datasets.make_swiss_roll(n_samples=n_samples,
                                        random_state=random_state)

        # Backup first axis, as it represents one of the underlying latent
        # variable we aim to recover
        y_pure = copy.deepcopy(x[:, 1])
        latents = np.vstack((y, y_pure)).T
        min = np.min(latents, axis=0)
        max = np.max(latents, axis=0)
        centers = (max + min) / 2
        ranges = np.abs(max - centers)
        upper_bounds = centers + np.array([.8, .3]) * ranges
        lower_bounds = centers - np.array([.8, .3]) * ranges

        self.interpolation_test = np.arange(x.shape[0])[
            (latents[:, 0] > lower_bounds[0])
            & (latents[:, 0] < upper_bounds[0])
            & (latents[:, 1] > lower_bounds[1])
            & (latents[:, 1] < upper_bounds[1])]

        # Normalize
        x = scipy.stats.zscore(x)

        # Get absolute distance from origin
        ab = np.abs(x[:, 1])
        sort = np.argsort(ab)

        # Take the sli_points points closest to origin
        # This is not used by the base class, but will be used by the SwissRoll
        # children class to remove a thin slice from the roll
        self.slice_idx = sort[0:sli_points]

        # Apply rotation  to achieve same variance on all axes
        x[:, 1] *= factor

        super().__init__(x, latents, split, split_ratio, random_state)

        # Latent variables are the coordinate on the "length" dimension and the color given by Sklearn
        # Both parametrize the intrinsic plane
        self.latents = self.targets.numpy()

        # Only keep one latent as target for compatibility with other datasets
        self.targets = self.targets[:, 0]


class SwissRoll(FullSwissRoll):
    """Swiss Roll class where part of the manifold is removed for testing.

    This is the dataset used in the GRAE paper."""

    def __init__(self, n_samples=SAMPLE, sli_points=250, split='none',
                 split_ratio=FIT_DEFAULT, random_state=SEED,
                 data_path=DEFAULT_PATH, test_mode='interpolation', factor=1):
        """Init.

        Args:
            n_samples(int, optional): Number of points to sample from the manifold.
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            sli_points(int, optional): Remove sli_points closest to origin on the "length" dimension and use them as
            the test split.
            data_path(str, optional): Unused. Only to share same signature with other datasets.
            test_mode(str, optional): 'slice' to remove a thin 'ribbon' of sli_points points removed from the middle of
            the manifold to test out of distribution generalization. 'interpolation' to remove a square in the
            plane of the roll and use it for testing.
            factor(int, optional): Stretch factor for the roll.
        """
        self.test_mode = test_mode

        super().__init__(n_samples, split, split_ratio=split_ratio,
                         random_state=random_state, sli_points=sli_points,
                         factor=factor)

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

        if self.test_mode == 'slice':
            test_idx = self.slice_idx
        elif self.test_mode == 'interpolation':
            test_idx = self.interpolation_test
        else:
            raise NotImplementedError()

        x_train, y_train, x_test, y_test = slice_3D(x, y, test_idx)

        if split == 'train':
            return torch.from_numpy(x_train), torch.from_numpy(y_train)
        else:
            return torch.from_numpy(x_test), torch.from_numpy(y_test)


class Torus(Surface):
    """Uniformly sampled torus. Can also sample toroidal helices."""

    def __init__(self, n_samples=SAMPLE, split='none', split_ratio=FIT_DEFAULT,
                 random_state=SEED, data_path=DEFAULT_PATH, main_r=3,
                 tube_r=1, helix=False, angle_offset=0, k=8):
        """Init.

        Args:
            n_samples(int, optional): Number of points to sample from the manifold.
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            data_path(str, optional): Unused. Only to share same signature with other datasets.
            main_r(float, optional): Distance from center of torus to the center of the tube.
            tube_r(float, optional): Radius of the tube.
            helix(bool, optional): Sample helix instead of full torus.
            k(int, optional): Number of curls in helix.
        """
        np.random.seed(random_state)
        x_list = list()
        y1_list = list()
        y2_list = list()
        n = 0

        while n < n_samples:
            v, w = np.random.uniform(size=(2, 5000))
            phi = 2 * np.pi * v

            if helix:
                theta = k * phi
            else:
                theta = 2 * np.pi * np.random.uniform(size=5000)

            c = main_r + tube_r * np.cos(theta)
            c1 = c * np.cos(phi + angle_offset)
            c2 = c * np.sin(phi + angle_offset)
            c3 = tube_r * np.sin(theta)
            candidates = np.vstack((c1, c2, c3)).T

            # Rejection sampling
            accepted = w < c / (main_r + tube_r)
            n += accepted.sum()
            x_list.append(candidates[accepted])
            y1_list.append(theta[accepted])
            y2_list.append(phi[accepted])

        x = np.vstack(x_list)[:n_samples]
        latents = np.vstack((np.hstack(y1_list), np.hstack(y2_list))).T[
                  :n_samples]

        super().__init__(x, latents, split, split_ratio, random_state)

        # Use main torus angle as latent variable
        self.latents = self.targets[:, 1].numpy()

        # Only keep one latent as target for compatibility with other datasets
        # Used as main coloring variable
        self.targets = self.targets[:, 1]


class ToroidalHelices(Surface):
    """Intertwined toroidal helices."""

    def __init__(self, n_samples=4000, n_helix=2, split='none',
                 split_ratio=FIT_DEFAULT, random_state=SEED,
                 data_path=DEFAULT_PATH, main_r=3, tube_r=1, k=8):
        """Init.

        Args:
            n_samples(int, optional): Number of points to sample per helix.
            n_helix(int, optional): Number of helices.
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            data_path(str, optional): Unused. Only to share same signature with other datasets.
            main_r(float, optional): Distance from center of torus to the center of the tube.
            tube_r(float, optional): Radius of the tube.
            k(int, optional): Number of curls in helix.
        """
        x_list = list()
        y_list = list()

        rotation = 2 * np.pi / (k * n_helix)

        for i in range(n_helix):
            helix = Torus(n_samples=n_samples, helix=True, k=k,
                          main_r=main_r, tube_r=tube_r, split='none',
                          angle_offset=i * rotation,
                          random_state=random_state + i)
            x, y = helix.numpy()
            y = np.vstack((np.full(shape=(n_samples,), fill_value=i), y)).T
            x_list.append(x)
            y_list.append(y)

        super().__init__(np.vstack(x_list), np.vstack(y_list),
                         split, split_ratio, random_state)

        # Use torus id and main angle as latents
        self.latents = self.targets.numpy()

        if n_helix > 1:
            # Use helix id if multiple helices
            self.targets = self.targets[:, 0]
        else:
            # If only one helix, use angle as target variable
            self.targets = self.targets[:, 1]


class ArtificialTree(BaseDataset):
    """High-dimensional artificial tree from the PHATE paper."""

    def __init__(self, n_dim=200, n_branch=10, branch_length=1000,
                 rand_multiplier=2, sigma=0,
                 split='none', split_ratio=FIT_DEFAULT, random_state=SEED,
                 data_path=DEFAULT_PATH):
        """Init.

        Args:
            n_dim(int, optional): Ambient space dimension.
            n_branch(int, optional): Number of branches to generate from main branch.
            branch_length(int, optional): Number of points in each branch.
            rand_multiplier(float, optional): Step between each point.
            sigma(float, optional): Variance of noise
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            data_path(str, optional): Unused. Only to share same signature with other datasets.
        """
        tree, branches = phate.tree.gen_dla(n_dim=n_dim, n_branch=n_branch,
                                            branch_length=branch_length,
                                            rand_multiplier=rand_multiplier,
                                            seed=random_state, sigma=sigma)
        point_id = np.tile(np.arange(branch_length), n_branch)

        super().__init__(tree, np.vstack((branches, point_id)).T,
                         split, split_ratio, random_state)
        self.latents = self.targets.numpy()
        self.targets = self.targets[:, 0]


class Mammoth(Surface):
    """Mammoth skeleton dataset. 10 k points in 3D.

    Credits to Smithsonian Institute, Andy Coenen & Adam Pearce.
    See their blog post at https://pair-code.github.io/understanding-umap/.
    """

    def __init__(self, split='none', split_ratio=FIT_DEFAULT,
                 random_state=SEED, data_path=DEFAULT_PATH):
        """Init.

        Args:
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            data_path(str, optional): Unused. Only to share same signature with other datasets.
        """
        self.url = 'https://raw.githubusercontent.com/PAIR-code/understanding-umap/master/raw_data/mammoth_3d.json'
        self.root = os.path.join(data_path, 'mammoth')

        if not os.path.exists(self.root):
            os.mkdir(self.root)
            self._download()

        x = np.load(os.path.join(self.root, 'x.npy'))
        y = np.arange(x.shape[0])

        super().__init__(x, y, split, split_ratio, random_state)

    def _download(self):
        """Download ans save dataset."""
        print('Downloading Mammoth dataset')
        results = requests.get(self.url).json()
        d = np.array(results)
        np.save(os.path.join(self.root, 'x'), d)


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
                 random_state=SEED, data_path=DEFAULT_PATH):
        """Init.

        Args:
            split(str, optional): Name of split. See BaseDataset.
            split_ratio(float, optional): Ratio of train split. See BaseDataset.
            random_state(int, optional): Random seed. See BaseDataset.
            data_path(str, optional): Unused. Only to share same signature with other datasets.
        """
        x, y = create_sphere_dataset(seed=random_state)

        super().__init__(x, y, split, split_ratio, random_state)
