"""PHATE, AE and GRAE model classes with sklearn inspired interface."""
import os

import torch
import torch.nn as nn
import numpy as np
import scipy

from grae.data.base_dataset import DEVICE
from grae.data.base_dataset import FromNumpyDataset
from grae.models import BaseModel
from grae.models.base_model import SEED
from grae.models.manifold_tools import PHATE, UMAP
from grae.models.torch_modules import AutoencoderModule, ConvAutoencoderModule

# Hyperparameters defaults
BATCH_SIZE = 128
LR = .0001
WEIGHT_DECAY = 0
EPOCHS = 200
HIDDEN_DIMS = (800, 400, 200)  # Default fully-connected dimensions
CONV_DIMS = [32, 64]  # Default conv channels
CONV_FC_DIMS = [400, 200]  # Default fully-connected dimensions after convs


class AE(BaseModel):
    """Vanilla Autoencoder model.

    Trained with Adam and MSE Loss.
    Model will infer from the data whether to use a fully FC or convolutional + FC architecture.
    """

    def __init__(self, *,
                 lr=LR,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 weight_decay=WEIGHT_DECAY,
                 random_state=SEED,
                 n_components=2,
                 hidden_dims=HIDDEN_DIMS,
                 conv_dims=CONV_DIMS,
                 conv_fc_dims=CONV_FC_DIMS,
                 noise=0,
                 patience=50,
                 data_val=None,
                 comet_exp=None,
                 write_path=''):
        """Init. Arguments specify the architecture of the encoder. Decoder will use the reversed architecture.

        Args:
            lr(float): Learning rate.
            epochs(int): Number of epochs for model training.
            batch_size(int): Mini-batch size.
            weight_decay(float): L2 penalty.
            random_state(int): To seed parameters and training routine for reproducible results.
            n_components(int): Bottleneck dimension.
            hidden_dims(List[int]): Number and size of fully connected layers for encoder. Do not specify the input
            layer or the bottleneck layer, since they are inferred from the data or from the n_components
            argument respectively. Decoder will use the same dimensions in reverse order. This argument is only used if
            provided samples are flat vectors.
            conv_dims(List[int]): Specify the number of convolutional layers. The int values specify the number of
            channels for each layer. This argument is only used if provided samples are images (i.e. 3D tensors)
            conv_fc_dims(List[int]): Number and size of fully connected layers following the conv_dims convolutionnal
            layer. No need to specify the bottleneck layer. This argument is only used if provided samples
            are images (i.e. 3D tensors)
            noise(float): Variance of the gaussian noise injected in the bottleneck before reconstruction.
            patience(int): Epochs with no validation MSE improvement before early stopping.
            data_val(BaseDataset): Split to validate MSE on for early stopping.
            comet_exp(Experiment): Comet experiment to log results.
            write_path(str): Where to write temp files.
        """
        self.random_state = random_state
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.fitted = False  # If model was fitted
        self.torch_module = None  # Will be initialized to the appropriate torch module when fit method is called
        self.optimizer = None  # Will be initialized to the appropriate optimizer when fit method is called
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss(reduction='mean')
        self.conv_dims = conv_dims
        self.conv_fc_dims = conv_fc_dims
        self.noise = noise
        self.comet_exp = comet_exp
        self.data_shape = None  # Shape of input data

        # Early stopping attributes
        self.data_val = data_val
        self.val_loader = None
        self.patience = patience
        self.current_loss_min = np.inf
        self.early_stopping_count = 0
        self.write_path = write_path

    def init_torch_module(self, data_shape, vae=False, sigmoid=False):
        """Infer autoencoder architecture (MLP or Convolutional + MLP) from data shape.

        Initialize torch module.

        Args:
            data_shape(tuple[int]): Shape of one sample.
            vae(bool): Make this architecture a VAE.
            sigmoid(bool): Apply sigmoid to decoder output.

        """
        # Infer input size from data. Initialize torch module and optimizer
        if len(data_shape) == 1:
            # Samples are flat vectors. MLP case
            input_size = data_shape[0]
            self.torch_module = AutoencoderModule(input_dim=input_size,
                                                  hidden_dims=self.hidden_dims,
                                                  z_dim=self.n_components,
                                                  noise=self.noise,
                                                  vae=vae,
                                                  sigmoid=sigmoid)
        elif len(data_shape) == 3:
            in_channel, height, width = data_shape
            #  Samples are 3D tensors (i.e. images). Convolutional case.
            self.torch_module = ConvAutoencoderModule(H=height,
                                                      W=width,
                                                      input_channel=in_channel,
                                                      channel_list=self.conv_dims,
                                                      hidden_dims=self.conv_fc_dims,
                                                      z_dim=self.n_components,
                                                      noise=self.noise,
                                                      vae=vae,
                                                      sigmoid=sigmoid)
        else:
            raise Exception(f'Invalid channel number. X has {len(data_shape)}')

        self.torch_module.to(DEVICE)

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """

        # Reproducibility
        torch.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Save data shape
        self.data_shape = x[0][0].shape

        # Fetch appropriate torch module
        if self.torch_module is None:
            self.init_torch_module(self.data_shape)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.torch_module.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        # Train AE
        # Training steps are decomposed as calls to specific methods that can be overriden by children class if need be
        self.torch_module.train()

        self.loader = self.get_loader(x)

        if self.data_val is not None:
            self.val_loader = self.get_loader(self.data_val)

        # Get first metrics
        self.log_metrics(0)

        for epoch in range(1, self.epochs + 1):
            # print(f'            Epoch {epoch}...')
            for batch in self.loader:
                self.optimizer.zero_grad()
                self.train_body(batch)
                self.optimizer.step()

            self.log_metrics(epoch)
            self.end_epoch(epoch)

            # Early stopping
            if self.early_stopping_count == self.patience:
                if self.comet_exp is not None:
                    self.comet_exp.log_metric('early_stopped',
                                              epoch - self.early_stopping_count)
                break

        # Load checkpoint if it exists
        checkpoint_path = os.path.join(self.write_path, 'checkpoint.pt')

        if os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
            os.remove(checkpoint_path)

    def get_loader(self, x):
        """Fetch data loader.

        Args:
            x(BaseDataset): Data to be wrapped in loader.

        Returns:
            torch.utils.data.DataLoader: Torch DataLoader for mini-batch training.

        """
        return torch.utils.data.DataLoader(x, batch_size=self.batch_size, shuffle=True)

    def train_body(self, batch):
        """Called in main training loop to update torch_module parameters.

        Args:
            batch(tuple[torch.Tensor]): Training batch.

        """
        data, _, idx = batch  # No need for labels. Training is unsupervised
        data = data.to(DEVICE)

        x_hat, z = self.torch_module(data)  # Forward pass
        self.compute_loss(data, x_hat, z, idx)

    def compute_loss(self, x, x_hat, z, idx):
        """Apply loss to update parameters following a forward pass.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        loss = self.criterion(x_hat, x)
        loss.backward()

    def end_epoch(self, epoch):
        """Method called at the end of every training epoch.

        Args:
            epoch(int): Current epoch.

        """
        pass

    def eval_MSE(self, loader):
        """Compute MSE on data.

        Args:
            loader(DataLoader): Dataset loader.

        Returns:
            float: MSE.

        """
        # Compute MSE over dataset in loader
        self.torch_module.eval()
        sum_loss = 0

        for batch in loader:
            data, _, idx = batch  # No need for labels. Training is unsupervised
            data = data.to(DEVICE)

            x_hat, z = self.torch_module(data)  # Forward pass
            sum_loss += data.shape[0] * self.criterion(data, x_hat).item()

        self.torch_module.train()

        return sum_loss / len(loader.dataset)  # Return average per observation

    def log_metrics(self, epoch):
        """Log metrics.

        Args:
            epoch(int): Current epoch.

        """
        self.log_metrics_train(epoch)
        self.log_metrics_val(epoch)

    def log_metrics_val(self, epoch):
        """Compute validation metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Validation loss
        if self.val_loader is not None:
            val_mse = self.eval_MSE(self.val_loader)

            if self.comet_exp is not None:
                with self.comet_exp.validate():
                    self.comet_exp.log_metric('MSE_loss', val_mse, epoch=epoch)

            if val_mse < self.current_loss_min:
                # If new min, update attributes and checkpoint model
                self.current_loss_min = val_mse
                self.early_stopping_count = 0
                self.save(os.path.join(self.write_path, 'checkpoint.pt'))
            else:
                self.early_stopping_count += 1

    def log_metrics_train(self, epoch):
        """Log train metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Train loss
        if self.comet_exp is not None:
            train_mse = self.eval_MSE(self.loader)
            with self.comet_exp.train():
                self.comet_exp.log_metric('MSE_loss', train_mse, epoch=epoch)

    def transform(self, x):
        """Transform data.

        Args:
            x(BaseDataset): Dataset to transform.
        Returns:
            ndarray: Embedding of x.

        """
        self.torch_module.eval()
        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                             shuffle=False)
        z = [self.torch_module.encoder(batch.to(DEVICE)).cpu().detach().numpy() for batch, _, _ in loader]
        return np.concatenate(z)

    def inverse_transform(self, x):
        """Take coordinates in the embedding space and invert them to the data space.

        Args:
            x(ndarray): Points in the embedded space with samples on the first axis.
        Returns:
            ndarray: Inverse (reconstruction) of x.

        """
        self.torch_module.eval()
        x = FromNumpyDataset(x)
        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                             shuffle=False)
        x_hat = [self.torch_module.decoder(batch.to(DEVICE)).cpu().detach().numpy()
                 for batch in loader]

        return np.concatenate(x_hat)

    def save(self, path):
        """Save state dict.

        Args:
            path(str): File path.

        """
        state = self.torch_module.state_dict()
        state['data_shape'] = self.data_shape
        torch.save(state, path)

    def load(self, path):
        """Load state dict.

        Args:
            path(str): File path.

        """
        state = torch.load(path)
        data_shape = state.pop('data_shape')

        if self.torch_module is None:
            self.init_torch_module(data_shape)

        self.torch_module.load_state_dict(state)


class GRAEBase(AE):
    """Standard GRAE class.

    AE with geometry regularization. The bottleneck is regularized to match an embedding precomputed by a manifold
    learning algorithm.
    """

    def __init__(self, *, embedder, embedder_params, lam=100, relax=False, **kwargs):
        """Init.

        Args:
            embedder(BaseModel): Manifold learning class constructor.
            embedder_params(dict): Parameters to pass to embedder.
            lam(float): Regularization factor.
            relax(bool): Use the lambda relaxation scheme. Set to false to use constant lambda throughout training.
            **kwargs: All other arguments with keys are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.lam_original = lam  # Needed to compute the lambda relaxation
        self.target_embedding = None  # To store the target embedding as computed by embedder
        self.relax = relax
        self.embedder = embedder(random_state=self.random_state,
                                 n_components=self.n_components,
                                 **embedder_params)  # To compute target embedding.

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        print('       Fitting GRAE...')
        print('           Fitting manifold learning embedding...')
        emb = scipy.stats.zscore(self.embedder.fit_transform(x))  # Normalize embedding
        self.target_embedding = torch.from_numpy(emb).float().to(DEVICE)

        print('           Fitting encoder & decoder...')
        super().fit(x)

    def compute_loss(self, x, x_hat, z, idx):
        """Compute torch-compatible geometric loss.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        if self.lam > 0:
            loss = self.criterion(x, x_hat) + self.lam * self.criterion(z, self.target_embedding[idx])
        else:
            loss = self.criterion(x, x_hat)

        loss.backward()

    def log_metrics_train(self, epoch):
        """Log train metrics to comet if comet experiment was set.

        Args:
            epoch(int): Current epoch.

        """
        if self.comet_exp is not None:

            # Compute MSE and Geometric Loss over train set
            self.torch_module.eval()
            sum_loss = 0
            sum_geo_loss = 0

            for batch in self.loader:
                data, _, idx = batch  # No need for labels. Training is unsupervised
                data = data.to(DEVICE)

                x_hat, z = self.torch_module(data)  # Forward pass
                sum_loss += data.shape[0] * self.criterion(data, x_hat).item()
                sum_geo_loss += data.shape[0] * self.criterion(z, self.target_embedding[idx]).item()

            with self.comet_exp.train():
                mse_loss = sum_loss / len(self.loader.dataset)
                geo_loss = sum_geo_loss / len(self.loader.dataset)
                self.comet_exp.log_metric('MSE_loss', mse_loss, epoch=epoch)
                self.comet_exp.log_metric('geo_loss', geo_loss, epoch=epoch)
                self.comet_exp.log_metric('GRAE_loss', mse_loss + self.lam * geo_loss, epoch=epoch)
                if self.lam * geo_loss > 0:
                    self.comet_exp.log_metric('geo_on_MSE', self.lam * geo_loss / mse_loss, epoch=epoch)

            self.torch_module.train()

    def end_epoch(self, epoch):
        """Method called at the end of every training epoch.

        Previously used to decay lambda according to the scheme described in the IEEE paper.

        Now using a scheme adapted to early stopping : turn off geometric regularization when reaching 50 % of patience

        Args:
            epoch(int): Current epoch.

        """
        if self.relax and self.lam > 0 and self.early_stopping_count == int(self.patience / 2):
            self.lam = 0  # Turn off constraint

            if self.comet_exp is not None:
                self.comet_exp.log_metric('relaxation', epoch, epoch=epoch)

        # Sigmoid shape that quickly drops from lam_original to 0 around 50 % of training epochs.
        # if self.relax:
        #     self.lam = (-self.lam_original * np.exp((epoch - (self.epochs / 2)) * 0.2)) / (
        #             1 + np.exp((epoch - (self.epochs / 2)) * 0.2)) \
        #                + self.lam_original


class GRAE(GRAEBase):
    """Standard GRAE class with PHATE-based geometric regularization.

    AE with geometry regularization. The bottleneck is regularized to match an embedding precomputed by the PHATE
    manifold learning algorithm.
    """

    def __init__(self, *, lam=100, knn=5, gamma=1, t='auto', relax=False, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            knn(int): knn argument of PHATE. Number of neighbors to consider in knn graph.
            t(int): Number of steps of the diffusion operator. Can also be set to 'auto' to select t according to the
            knee point in the Von Neumann Entropy of the diffusion operator
            gamma(float): Informational distance.
            relax(bool): Use the lambda relaxation scheme. Set to false to use constant lambda throughout training.
            **kwargs: All other kehyword arguments are passed to the GRAEBase parent class.
        """
        super().__init__(lam=lam,
                         relax=relax,
                         embedder=PHATE,
                         embedder_params=dict(knn=knn,
                                              t=t,
                                              gamma=gamma,
                                              verbose=0,
                                              n_jobs=-1),
                         **kwargs)


class GRAE_R(GRAEBase):
    """Relaxed GRAE class with PHATE-based geometric regularization.
    """

    def __init__(self, *, lam=10, knn=5, gamma=1, t='auto', **kwargs):
        """Init.

        Args:
            lam(float): Initial regularization factor. Will be relaxed throughout training.
            knn(int): knn argument of PHATE. Number of neighbors to consider in knn graph.
            t(int): Number of steps of the diffusion operator. Can also be set to 'auto' to select t according to the
            knee point in the Von Neumann Entropy of the diffusion operator
            gamma(float): Informational distance.
            **kwargs: All other keyword arguments are passed to the GRAEBase parent class.
        """
        super().__init__(lam=lam,
                         relax=True,
                         embedder=PHATE,
                         embedder_params=dict(knn=knn,
                                              t=t,
                                              gamma=gamma,
                                              verbose=0,
                                              n_jobs=-1),
                         **kwargs)


class SmallGRAE(GRAE):
    """GRAE class with fixed small geometric regularization factor."""

    def __init__(self, *, knn=5, t='auto', **kwargs):
        """Init.

        Args:
            knn(int): knn argument of PHATE. Number of neighbors to consider in knn graph.
            t(int): Number of steps of the diffusion operator. Can also be set to 'auto' to select t according to the
            knee point in the Von Neumann Entropy of the diffusion operator
            **kwargs: All other arguments with keys are passed to the GRAE parent class.
        """
        super().__init__(lam=.1, relax=False, knn=knn, t=t, **kwargs)


class LargeGRAE(GRAE):
    """GRAE class with fixed large geometric regularization factor."""

    def __init__(self, *, knn=5, t='auto', **kwargs):
        """Init.

        Args:
            knn(int): knn argument of PHATE. Number of neighbors to consider in knn graph.
            t(int): Number of steps of the diffusion operator. Can also be set to 'auto' to select t according to the
            knee point in the Von Neumann Entropy of the diffusion operator
            **kwargs: All other arguments with keys are passed to the GRAE parent class.
        """
        super().__init__(lam=100, relax=False, knn=knn, t=t, **kwargs)


class GRAEUMAP(GRAEBase):
    """GRAE with UMAP regularization."""

    def __init__(self, *, lam=100, n_neighbors=15, min_dist=.1, relax=False, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            n_neighbors(int): The size of local neighborhood (in terms of number of neighboring sample points) used for
            manifold approximation.
            min_dist(float):  The effective minimum distance between embedded points.
            relax(bool): Use the lambda relaxation scheme. Set to false to use constant lambda throughout training.
            **kwargs: All other arguments with keys are passed to the GRAEBase parent class.
        """
        super().__init__(lam=lam,
                         embedder=UMAP,
                         embedder_params=dict(n_neighbors=n_neighbors, min_dist=min_dist),
                         relax=relax,
                         **kwargs)


class GRAEUMAP_R(GRAEBase):
    """Relaxed GRAE with UMAP regularization."""

    def __init__(self, *, lam=10, n_neighbors=15, min_dist=.1, **kwargs):
        """Init.

        Args:
            lam(float): Initial regularization factor. Will be relaxed throughout training.
            n_neighbors(int): The size of local neighborhood (in terms of number of neighboring sample points) used for
            manifold approximation.
            min_dist(float):  The effective minimum distance between embedded points.
            relax(bool): Use the lambda relaxation scheme. Set to false to use constant lambda throughout training.
            **kwargs: All other arguments with keys are passed to the GRAEBase parent class.
        """
        super().__init__(lam=lam,
                         embedder=UMAP,
                         embedder_params=dict(n_neighbors=n_neighbors, min_dist=min_dist),
                         relax=True,
                         **kwargs)