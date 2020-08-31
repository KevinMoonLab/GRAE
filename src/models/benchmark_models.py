"""Other models to compare GRAE."""
import torch
import umap
from sklearn.neighbors import NearestNeighbors
import numpy as np

from src.models.models import BaseModel, AE
from src.models.topo import TopoAELoss, compute_distance_matrix
from src.data.base import device


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
