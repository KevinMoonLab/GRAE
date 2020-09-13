"""Other models to compare GRAE."""
import torch
import umap
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy

from src.models.models import BaseModel, AE, GRAE
from src.models.topo import TopoAELoss, compute_distance_matrix
from src.data.base import device

from src.models import Diffusion as df
from pydiffmap import diffusion_map as dm
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import kneighbors_graph


class UMAP(umap.UMAP, BaseModel):
    """Thin wrapper for UMAP to work with torch datasets."""

    def fit(self, X):
        x, _ = X.numpy()
        super().fit(x)

    def fit_transform(self, X):
        x, _ = X.numpy()
        super().fit(x)
        return super().transform(x)

    def transform(self, X):
        x, _ = X.numpy()
        return super().transform(x)

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

class GRAEUMAP(GRAE):
    def __init__(self, *, lam=100, embedder_args=dict(), **kwargs):
        super().__init__(lam=lam,
                         embedder=UMAP,
                         embedder_args=embedder_args,
                         relax=True,
                         **kwargs)

class TopoAE(AE):
    """AE with topological loss. See topo.py"""

    def __init__(self, *, lam=100, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.topo_loss = TopoAELoss()

    def apply_loss(self, x, x_hat, z, idx):
        loss = self.criterion(x, x_hat) + self.lam * self.topo_loss(x, z)

        loss.backward()


class EAERMargin(AE):
    def __init__(self, *, lam=1, n_neighbors=10, margin=1, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.n_neighbors = n_neighbors
        self.margin = margin

    def fit(self, x):
        x_np, _ = x.numpy()
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(x_np)

        self.knn_graph = nbrs.kneighbors_graph()

        # Use whole dataset as batch, as in the paper
        self.batch_size = len(x)

        super().fit(x)

    def apply_loss(self, x, x_hat, z, idx):
        print(self.lr)
        print(self.batch_size)
        if self.lam > 0:
            batch_d = compute_distance_matrix(z)
            is_nb = torch.from_numpy(self.knn_graph[np.ix_(idx, idx)].toarray()).to(device)

            # Dummy zeros
            zero = torch.zeros(batch_d.shape).to(device)

            d = is_nb * batch_d + (1 - is_nb) * (torch.max(zero, self.margin - batch_d)) ** 2

            margin_loss = torch.sum(d)

            loss = self.criterion(x, x_hat) + self.lam * margin_loss
        else:
            loss = self.criterion(x, x_hat)

        loss.backward()


class DiffusionNet(AE):
    def __init__(self, *, lam=100, n_neighbors=100, alpha=1, epsilon='bgh', subsample=None, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.epsilon = epsilon
        self.subsample = subsample

    def fit(self, x):
        # DiffusionNet do not support mini-batches. Subspample data if needed to fit in memory
        if self.subsample is not None:
            x = x.subset(self.subsample, random_state=self.random_state)

        # Use whole dataset as batch, as in the paper
        self.batch_size = len(x)

        x_np, _ = x.numpy()

        # K_mat = df.ComputeLBAffinity(x_np, self.n_neighbors, sig=0.1)   # Laplace-Beltrami affinity: D^-1 * K * D^-1
        # self.P  = torch.from_numpy(df.makeRowStoch(K_mat)).to(device)                     # markov matrix 
        # Evectors, Evalues = df.Diffusion(K_mat, 
        #                                             nEigenVals = self.n_components+1)  # eigenvalues and eigenvectors

        neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=self.n_components,
                                              alpha=self.alpha,
                                              epsilon=self.epsilon,
                                              k=self.n_neighbors,
                                              neighbor_params=neighbor_params)
        dmap = mydmap.fit_transform(x_np)
        self.z = torch.tensor(dmap).float().to(device)

        self.Evectors = torch.from_numpy(mydmap.evecs).float().to(device)
        self.Evalues = torch.from_numpy(mydmap.evals).float().to(device)

        # Potential matrix sparse form
        P = scipy.sparse.coo_matrix(mydmap.L.todense())
        values = P.data
        indices = np.vstack((P.row, P.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        self.P = torch.sparse.FloatTensor(i, v).float().to(device)

        # Identity matrix sparse 
        I_n = scipy.sparse.coo_matrix(np.eye(self.batch_size))
        values = I_n.data
        indices = np.vstack((I_n.row, I_n.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        self.I_t = torch.sparse.FloatTensor(i, v).float().to(device)
        super().fit(x)

    def apply_loss(self, x, x_hat, z, idx):
        if self.lam > 0:

            rec_loss = self.criterion(x, x_hat)
            coord_loss = self.lam * self.criterion(z, self.z[idx])
            Ev_loss = (torch.mean(torch.pow(torch.mm((self.P.to_dense() - self.Evalues[0] *
                                                                            self.I_t.to_dense()),
                                                                           z[:, 0].view(self.batch_size, 1)),
                                                                  2)) + torch.mean(
                torch.pow(torch.mm((self.P.to_dense() - self.Evalues[1] *
                                    self.I_t.to_dense()),
                                   z[:, 1].view(self.batch_size, 1)), 2)))

            loss = rec_loss + coord_loss + self.lam * Ev_loss
            # print(rec_loss)
            # print(coord_loss)
            # print(Ev_loss)
        else:

            loss = self.criterion(x, x_hat)

        loss.backward()


class LaplacianAE(AE):
    def __init__(self, *, lam=1, n_neighbors=100, alpha=0, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.n_neighbors = n_neighbors
        self.alpha = alpha

    def fit(self, x):
        x_np, _ = x.numpy()

        D = kneighbors_graph(x_np, mode='distance',
                             n_neighbors=self.n_neighbors)

        W = np.multiply(np.exp(-D.toarray() / 1), D.toarray() != 0)
        # W = scipy.sparse.csr_matrix(W)

        self.W = torch.tensor(W).float().to(device)

        # Use whole dataset as batch, as in the paper
        self.batch_size = len(x)
        super().fit(x)

    def coord_loss(self, z):
        e = torch.pow(compute_distance_matrix(z), 2)
        # print(e.shape)
        L = torch.mul(self.W, e)
        # print(L.shape)
        return torch.sum(L)

    def apply_loss(self, x, x_hat, z, idx):

        if self.lam > 0:

            rec_loss = self.criterion(x, x_hat)
            # print('Reconstruction')
            # print(rec_loss.item())

            loss = 0 * rec_loss + 1e6 * self.coord_loss(z)
            # print('Total loss')
            # print(loss.item())

        else:

            loss = self.criterion(x, x_hat)

        loss.backward()
