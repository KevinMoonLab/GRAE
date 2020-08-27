"""Other models to compare GRAE."""
import umap

from src.models.models import BaseModel, AE
from src.models.topo import TopoAELoss


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


class TopoAE(AE):
    """AE with topological loss. See topo.py"""

    def __init__(self, *, lam=1000, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.topo_loss = TopoAELoss()

    def apply_loss(self, x, x_hat, z, idx):
        loss = self.criterion(x, x_hat) + self.lam * self.topo_loss(x, z)

        loss.backward()
