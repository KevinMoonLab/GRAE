"""Other models to compare GRAE."""

import numpy as np
import scipy
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from pydiffmap import diffusion_map as dm

from grae.models.grae_models import AE
from grae.models.external_tools.topological_loss import TopoAELoss, compute_distance_matrix
from grae.data.base_dataset import DEVICE


class TopoAE(AE):
    """Topological Autoencoder.

    From the paper of the same name. See https://arxiv.org/abs/1906.00722.

    See external_tools/topological_loss.py for the loss definition. Adapted from their source code.
    """

    def __init__(self, *, lam=100, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.topo_loss = TopoAELoss()

    def compute_loss(self, x, x_hat, z, idx):
        """Compute topological loss over a batch.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        loss = self.criterion(x, x_hat) + self.lam * self.topo_loss(x, z)

        loss.backward()


class EAERMargin(AE):
    """AE with margin-based regularization in the latent space.

    As presented in the EAER paper. See https://link.springer.com/chapter/10.1007/978-3-642-40994-3_14

    Note : The algorithm was adapted to support mini-batch training and SGD.
    """

    def __init__(self, *, lam=100, n_neighbors=10, margin=1, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            n_neighbors(int): The size of local neighborhood used to build the neighborhood graph.
            margin(float):  Margin used for the max-margin loss.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.n_neighbors = n_neighbors
        self.margin = margin
        self.knn_graph = None  # Will store the neighborhood graph of the data

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        x_np, _ = x.numpy()

        # Determine neighborhood parameters
        x_np, _ = x.numpy()
        if x_np.shape[1] > 100:
            print('Computing PCA before knn search...')
            x_np = PCA(n_components=100).fit_transform(x_np)

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto').fit(x_np)

        self.knn_graph = nbrs.kneighbors_graph()

        super().fit(x)

    def compute_loss(self, x, x_hat, z, idx):
        """Compute max-margin loss over a batch.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        if self.lam > 0:
            batch_d = compute_distance_matrix(z)
            is_nb = torch.from_numpy(self.knn_graph[np.ix_(idx, idx)].toarray()).to(DEVICE)

            clipped_dist = torch.clamp(input=self.margin - batch_d, min=0)

            d = is_nb * batch_d + (1 - is_nb) * clipped_dist ** 2

            margin_loss = torch.sum(d)

            loss = self.criterion(x, x_hat) + self.lam * margin_loss
        else:
            loss = self.criterion(x, x_hat)

        loss.backward()


class DiffusionNet(AE):
    """Diffusion nets.

    As presented in https://arxiv.org/abs/1506.07840

    Note: Subsampling was required to run this model on our benchmarks.

    """

    def __init__(self, *, lam=100, eta=100, n_neighbors=100, alpha=1, epsilon='bgh_generous', subsample=None, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor for the coordinate constraint.
            eta(float): Regularization factor for the EV constraint.
            n_neighbors(int): The size of local neighborhood used to build the neighborhood graph.
            alpha(float): Exponent to be used for the left normalization in constructing the diffusion map.
            epsilon(Any):  Method for choosing the epsilon. See scikit-learn NearestNeighbors class for details.
            subsample(int): Number of points to sample from the dataset before fitting.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.epsilon = epsilon
        self.subsample = subsample
        self.eta = eta

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        # DiffusionNet do not support mini-batches. Subsample data if needed to fit in memory
        if self.subsample is not None:
            x = x.random_subset(self.subsample, random_state=self.random_state)

        # Use whole dataset (after possible subsampling) as batch, as in the paper
        self.batch_size = len(x)

        x_np, _ = x.numpy()

        # Reduce dimensionality for faster kernel computations. We do the same with PHATE and UMAP.
        if x_np.shape[1] > 100 and x_np.shape[0] > 1000:
            print('Computing PCA before running DM...')
            x_np = PCA(n_components=100).fit_transform(x_np)

        neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=self.n_components,
                                              alpha=self.alpha,
                                              epsilon=self.epsilon,
                                              k=self.n_neighbors,
                                              neighbor_params=neighbor_params)
        dmap = mydmap.fit_transform(x_np)

        self.z = torch.tensor(dmap).float().to(DEVICE)

        self.Evectors = torch.from_numpy(mydmap.evecs).float().to(DEVICE)
        self.Evalues = torch.from_numpy(mydmap.evals).float().to(DEVICE)

        # Potential matrix sparse form
        P = scipy.sparse.coo_matrix(mydmap.L.todense())
        values = P.data
        indices = np.vstack((P.row, P.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        self.P = torch.sparse.FloatTensor(i, v).float().to(DEVICE)

        # Identity matrix sparse 
        I_n = scipy.sparse.coo_matrix(np.eye(self.batch_size))
        values = I_n.data
        indices = np.vstack((I_n.row, I_n.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        self.I_t = torch.sparse.FloatTensor(i, v).float().to(DEVICE)
        super().fit(x)

    def compute_loss(self, x, x_hat, z, idx):
        """Compute diffusion-based loss.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        rec_loss = self.criterion(x, x_hat)
        coord_loss = self.criterion(z, self.z[idx])
        Ev_loss = (torch.mean(torch.pow(torch.mm((self.P.to_dense() - self.Evalues[0] *
                                                  self.I_t.to_dense()),
                                                 z[:, 0].view(self.batch_size, 1)),
                                        2)) + torch.mean(
            torch.pow(torch.mm((self.P.to_dense() - self.Evalues[1] *
                                self.I_t.to_dense()),
                               z[:, 1].view(self.batch_size, 1)), 2)))

        loss = rec_loss + self.lam * coord_loss + self.eta * Ev_loss

        loss.backward()
