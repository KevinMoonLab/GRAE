"""PHATE, AE and GRAE model classes with sklearn inspired interface."""

import torch
import torch.nn as nn
import numpy as np
import phate
import scipy

from src.data.base_dataset import DEVICE
from src.data.base_dataset import FromNumpyDataset
from src.models import BaseModel
from src.models.external_tools.procrustes import procrustes
from src.models.torch_modules import AutoencoderModule, ConvAutoencoderModule

# Hyperparameters defaults
SEED = 42
BATCH_SIZE = 128
LR = .0001
WEIGHT_DECAY = 1
EPOCHS = 200
HIDDEN_DIMS = (800, 400, 200)  # Default fully-connected dimensions
CONV_DIMS = [32, 64]  # Default conv channels
CONV_FC_DIMS = [400, 200]  # Default fully-connected dimensions after convs
PROC_THRESHOLD = 20000  # Procrustes threshold (see PHATE)


class PHATE(phate.PHATE, BaseModel):
    """Wrapper for PHATE to work with torch datasets.

    Also add procrustes transform when dealing with large datasets for improved scalability.
    """

    def __init__(self, proc_threshold=PROC_THRESHOLD, procrustes_batches_size=1000, procrustes_lm=1000, **kwargs):
        """Init.

        Args:
            proc_threshold(int): Threshold beyond which PHATE is computed over mini-batches of the data and batches are
            realigned with procrustes. Otherwise, vanilla PHATE is used.
            procrustes_batches_size(int): Batch size of procrustes approach.
            procrustes_lm (int): Number of anchor points present in all batches. Used as a reference for the procrustes
            transform.
            **kwargs: Any remaining keyword arguments are passed to the PHATE model.
        """
        self.proc_threshold = proc_threshold
        self.procrustes_batch_size = procrustes_batches_size
        self.procrustes_lm = procrustes_lm
        super().__init__(**kwargs)

    def fit_transform(self, x):
        """Fit model and transform data.

        Overrides PHATE fit_transform method on datasets larger than self.proc_threshold to compute PHATE over
        mini-batches with procrustes realignment.

        Args:
            x(BaseDataset): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x.

        """
        x, _ = x.numpy()

        if x.shape[0] < self.proc_threshold:
            result = super().fit_transform(x)
        else:
            print('            Fitting procrustes...')
            result = self.fit_transform_procrustes(x)
        return result

    def fit_transform_procrustes(self, x):
        """Fit model and transform data for larger datasets.

        If dataset has more than self.proc_threshold samples, then compute PHATE over
        mini-batches. In each batch, add self.procrustes_lm samples (which are the same for all batches),
        which can be used to compute a  procrustes transform to roughly align all batches in a coherent manner.
        This last step is required since PHATE can lead to embeddings with different rotations or reflections
        depending on the batch.

        Args:
            x(BaseDataset): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x, which is the union of all batches aligned with procrustes.

        """
        lm_points = x[:self.procrustes_lm, :]  # Reference points included in all batches
        initial_embedding = super().fit_transform(lm_points)
        result = [initial_embedding]
        remaining_x = x[self.procrustes_lm:, :]
        while len(remaining_x) != 0:
            if len(remaining_x) >= self.procrustes_batch_size:
                new_points = remaining_x[:self.procrustes_batch_size, :]
                remaining_x = np.delete(remaining_x,
                                        np.arange(self.procrustes_batch_size),
                                        axis=0)
            else:
                new_points = remaining_x
                remaining_x = np.delete(remaining_x,
                                        np.arange(len(remaining_x)),
                                        axis=0)

            subsetx = np.vstack((lm_points, new_points))
            subset_embedding = super().fit_transform(subsetx)

            d, Z, tform = procrustes(initial_embedding,
                                     subset_embedding[:self.procrustes_lm, :])

            subset_embedding_transformed = np.dot(
                subset_embedding[self.procrustes_lm:, :],
                tform['rotation']) + tform['translation']

            result.append(subset_embedding_transformed)
        return np.vstack(result)


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
                 noise=0):
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
        self.criterion = nn.MSELoss(reduction='sum')
        self.conv_dims = conv_dims
        self.conv_fc_dims = conv_fc_dims
        self.noise = noise
        self.comet_exp = None
        self.x_train = None

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        self.x_train = x

        # Reproducibility
        torch.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Fetch appropriate torch module
        if self.torch_module is None:
            # Infer input size from data. Initialize torch module and optimizer
            if len(x[0][0].shape) == 1:
                # Samples are flat vectors. FC case
                input_size = x[0][0].shape[0]
                self.torch_module = AutoencoderModule(input_dim=input_size,
                                                      hidden_dims=self.hidden_dims,
                                                      z_dim=self.n_components,
                                                      noise=self.noise)
            elif len(x[0][0].shape) == 3:
                in_channel, height, width = x[0][0].shape
                #  Samples are 3D tensors (i.e. images). Convolutional case.
                self.torch_module = ConvAutoencoderModule(H=height,
                                                          W=width,
                                                          input_channel=in_channel,
                                                          channel_list=self.conv_dims,
                                                          hidden_dims=self.conv_fc_dims,
                                                          z_dim=self.n_components,
                                                          noise=self.noise)
            else:
                raise Exception(f'Invalid channel number. X has {len(x[0][0].shape)}')

        # Optimizer
        self.optimizer = torch.optim.Adam(self.torch_module.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        # Train AE
        # Training steps are decomposed as calls to specific methods that can be overriden by children class if need be
        self.torch_module.to(DEVICE)
        self.torch_module.train()

        self.loader = self.get_loader(x)

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
        loss = self.criterion(x, x_hat)
        loss.backward()

    def end_epoch(self, epoch):
        """Method called at the end of every training epoch.

        Args:
            epoch(int): Current epoch.

        """
        pass

    def log_metrics(self, epoch):
        """Log metrics to comet if comet experiment was set.

        Args:
            epoch(int): Current epoch.

        """
        if self.comet_exp is not None:

            # Compute MSE over train set
            self.torch_module.eval()
            sum_loss = 0

            for batch in self.loader:
                data, _, idx = batch  # No need for labels. Training is unsupervised
                data = data.to(DEVICE)

                x_hat, z = self.torch_module(data)  # Forward pass
                sum_loss += self.criterion(data, x_hat).item()

            with self.comet_exp.train():
                self.comet_exp.log_metric('MSE_loss', sum_loss/len(self.loader.dataset), epoch=epoch)

            self.torch_module.train()

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


class GRAEBase(AE):
    """Standard GRAE class.

    AE with geometry regularization. The bottleneck is regularized to match an embedding precomputed by a manifold
    learning algorithm.
    """

    def __init__(self, *, embedder, embedder_params, lam=100, relax=True, **kwargs):
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

    def log_metrics(self, epoch):
        """Log metrics to comet if comet experiment was set.

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
                sum_loss += self.criterion(data, x_hat).item()
                sum_geo_loss += self.criterion(z, self.target_embedding[idx]).item()

            with self.comet_exp.train():
                mse_loss = sum_loss/len(self.loader.dataset)
                geo_loss = sum_geo_loss/len(self.loader.dataset)
                self.comet_exp.log_metric('MSE_loss', mse_loss, epoch=epoch)
                self.comet_exp.log_metric('geo_loss', geo_loss, epoch=epoch)
                self.comet_exp.log_metric('GRAE_loss', mse_loss + self.lam * geo_loss, epoch=epoch)
                if self.lam * geo_loss > 0:
                    self.comet_exp.log_metric('geo_on_MSE', self.lam * geo_loss/mse_loss, epoch=epoch)

            self.torch_module.train()

    def end_epoch(self, epoch):
        """Method called at the end of every training epoch.

        Used here to decay lambda according to the scheme described in the paper.

        Args:
            epoch(int): Current epoch.

        """
        # Sigmoid shape that quickly drops from lam_original to 0 around 50 % of training epochs.
        if self.relax:
            self.lam = (-self.lam_original * np.exp((epoch - (self.epochs / 2)) * 0.2)) / (
                    1 + np.exp((epoch - (self.epochs / 2)) * 0.2)) \
                       + self.lam_original


class GRAE(GRAEBase):
    """Standard GRAE class with PHATE-based geometric regularization.

    AE with geometry regularization. The bottleneck is regularized to match an embedding precomputed by the PHATE
    manifold learning algorithm.
    """

    def __init__(self, *, lam=100, knn=5, t='auto', relax=True, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            knn(int): knn argument of PHATE. Number of neighbors to consider in knn graph.
            t(int): Number of steps of the diffusion operator. Can also be set to 'auto' to select t according to the
            knee point in the Von Neumann Entropy of the diffusion operator
            relax(bool): Use the lambda relaxation scheme. Set to false to use constant lambda throughout training.
            **kwargs: All other kehyword arguments are passed to the GRAEBase parent class.
        """
        super().__init__(lam=lam,
                         relax=relax,
                         embedder=PHATE,
                         embedder_params=dict(knn=knn,
                                              t=t,
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
