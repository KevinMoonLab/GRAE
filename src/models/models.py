"""Model classes with sklearn inspired interface."""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import phate
import umap
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error

from src.data.base import device
from src.data.base import NumpyDataset
from src.models.torch_modules import AutoencoderModule, ConvAutoencoderModule

# Hyperparameters defaults
SEED = 42
BATCH_SIZE = 128
LR = .0001
WEIGHT_DECAY = 1
EPOCHS = 10000
HIDDEN_DIMS = (800, 400, 200)  # Default hidden MLP dimensions
CONV_DIMS = [32, 64]
CONV_FC_DIMS = [400, 200]


class BaseModel:
    """All models should subclass BaseModel."""

    def fit(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        raise NotImplementedError()

    def fit_plot(self, X, cmap='jet', s=1):
        z = self.fit_transform(X)
        y = X.targets.numpy()

        if z.shape[1] != 2:
            raise Exception('Can only plot 2D embeddings.')

        plt.scatter(*z.T, c=y, cmap=cmap, s=s)
        plt.show()

        return z


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


class PHATE(phate.PHATE, BaseModel):
    """Thin wrapper for PHATE to work with torch datasets."""

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


class AE(BaseModel):
    """Autoencoder model."""

    def __init__(self, *, lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY,
                 random_state=SEED, n_components=2, hidden_dims=HIDDEN_DIMS,
                 conv_dims=CONV_DIMS, conv_fc_dims=CONV_FC_DIMS):
        self.random_state = random_state
        self.n_components = n_components
        self.hidden_dims = hidden_dims  # List of dimensions of the hidden layers in the encoder (decoder will use
        # the inverse architecture
        self.fitted = False
        self.torch_module = None
        self.optimizer = None

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss(reduction='sum')
        self.conv_dims = conv_dims
        self.conv_fc_dims = conv_fc_dims

    def fit(self, X):
        # Reproducibility
        torch.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if self.fitted:
            raise Exception('Cannot fit a second time.')

        # Infer input size from data. Initialize torch module and optimizer
        if len(X[0][0].shape) == 1:
            # Linear case
            input_size = X[0][0].shape[0]
            self.torch_module = AutoencoderModule(input_dim=input_size,
                                                  hidden_dims=self.hidden_dims,
                                                  z_dim=self.n_components)
        elif len(X[0][0].shape) == 3:
            in_channel, height, width = X[0][0].shape
            #  Convolutionnal case
            self.torch_module = ConvAutoencoderModule(H=height,
                                                      W=width,
                                                      input_channel=in_channel,
                                                      channel_list=self.conv_dims,
                                                      hidden_dims=self.conv_fc_dims,
                                                      z_dim=self.n_components)
        else:
            raise Exception(f'Invalid channel number. X has {len(X[0][0].shape)}')

        self.optimizer = torch.optim.Adam(self.torch_module.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        # Train AE
        self.torch_module.to(device)
        self.torch_module.train()

        self.loader = self.get_loader(X)

        for epoch in range(self.epochs):
            print(f'            Epoch {epoch}...')
            for batch in self.loader:
                self.optimizer.zero_grad()
                self.train_body(batch)
                self.optimizer.step()

            self.end_epoch(epoch)

    def get_loader(self, X):
        return torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=True)

    def train_body(self, batch):
        data, _, idx = batch
        data = data.to(device)

        x_hat, z = self.torch_module(data)
        self.apply_loss(data, x_hat, z, idx)

    def apply_loss(self, x, x_hat, z, idx):
        # Standard AE does not use latents nor sample indices to compute loss,
        # but both will be relied on by children class.
        loss = self.criterion(x, x_hat)
        loss.backward()

    def end_epoch(self, epoch):
        pass

    def transform(self, X):
        self.torch_module.eval()
        loader = torch.utils.data.DataLoader(X, batch_size=self.batch_size,
                                             shuffle=False)
        z = [self.torch_module.encoder(batch.to(device)).cpu().detach().numpy() for batch, _, _ in loader]
        return np.concatenate(z)

    def inverse_transform(self, z):
        self.torch_module.eval()
        z = NumpyDataset(z)
        loader = torch.utils.data.DataLoader(z, batch_size=self.batch_size,
                                             shuffle=False)
        x_hat = [self.torch_module.decoder(batch.to(device)).cpu().detach().numpy()
                 for batch in loader]

        return np.concatenate(x_hat)

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

    def score_reconstruction(self, X):
        n = len(X)
        x_hat = self.reconstruct(X)
        x, _ = X.numpy()
        return mean_squared_error(x.reshape((n, -1)), x_hat.reshape((n, -1)))


class GRAE(AE):
    """Standard GRAE class."""

    def __init__(self, *, lam=10, drop_lam=None, embedder=PHATE, embedder_args=dict(), **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.embedder = embedder(**embedder_args,
                                 random_state=self.random_state, n_components=self.n_components)
        self.z = None
        self.drop_lam = drop_lam

    def fit(self, X):
        # Find manifold learning embedding
        emb = scipy.stats.zscore(self.embedder.fit_transform(X))
        self.z = torch.from_numpy(emb).float().to(device)

        super().fit(X)

    def apply_loss(self, x, x_hat, z, idx):
        if self.lam > 0:
            loss = self.criterion(x, x_hat) + self.lam * self.criterion(z, self.z[idx])
        else:
            loss = self.criterion(x, x_hat)

        loss.backward()

    def end_epoch(self, epoch):
        # Soft GRAE (turn off geometric loss after drop_lam epochs)
        if self.drop_lam is not None and epoch == self.drop_lam - 1:
            self.lam = 0


class SGRAE(AE):
    """Siamese variant of GRAE."""

    def __init__(self, *, lam=10, PHATE_args={}, drop_lam=None, **kwargs):
        super().__init__(**kwargs)
        self.dist_calculator = PHATE(**PHATE_args,
                                     random_state=self.random_state, n_components=self.n_components)
        self.a_mx = None
        self.d_mx = None
        self.lam = lam
        self.drop_lam = drop_lam

    def fit(self, X):
        # Fit PHATE
        self.dist_calculator.fit(X)

        x, _ = X.numpy()

        # Distance matrix based on potential distances
        self.d_mx = torch.from_numpy(squareform(pdist(self.dist_calculator.diff_potential))).to(device)

        super().fit(X)

    def apply_loss(self, x, x_hat, z, idx):

        if self.lam > 0:
            half = x.shape[0] // 2

            batch_1 = idx[:half]
            batch_2 = idx[half:2 * half]

            d_z = torch.norm(z[:half] - z[half:], p=2, dim=1)

            d = (d_z - self.d_mx[batch_1, batch_2]) ** 2

            geo_loss = torch.sum(d)

            loss = self.criterion(x, x_hat) + self.lam * geo_loss
        else:
            loss = self.criterion(x, x_hat)

        loss.backward()

    def end_epoch(self, epoch):
        # Soft GRAE (turn off geometric loss after drop_lam epochs)
        if self.drop_lam is not None and epoch == self.drop_lam - 1:
            self.lam = 0


class SoftGRAE(GRAE):
    """GRAE class where the geometric regularization is dropped after 50 % of epochs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drop_lam = self.epochs // 2
