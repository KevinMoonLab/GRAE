"""Model classes with sklearn inspired interface."""
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import phate
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
EPOCHS = 200
HIDDEN_DIMS = (800, 400, 200)  # Default fully-connected dimensions
CONV_DIMS = [32, 64]  # Default conv channels
CONV_FC_DIMS = [400, 200]  # Default fully-connected dimensions after convs
PROC_THRESHOLD = 20000  # Procrustes threshold


class BaseModel:
    """All models should subclass BaseModel."""

    def fit(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        raise NotImplementedError()

    def fit_plot(self, X, cmap='jet', s=1, fit=True):
        if fit:
            z = self.fit_transform(X)
        else:
            z = self.transform(X)

        y = X.targets.numpy()

        if z.shape[1] != 2:
            raise Exception('Can only plot 2D embeddings.')

        plt.scatter(*z.T, c=y, cmap=cmap, s=s)
        plt.show()

        return z

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

    def score(self, X, split_name):
        n = len(X)

        start = time.time()
        z = self.transform(X)
        stop = time.time()

        transform_time = stop - start

        start = time.time()
        x_hat = self.inverse_transform(z)
        stop = time.time()

        rec_time = stop - start

        x, _ = X.numpy()
        MSE = mean_squared_error(x.reshape((n, -1)), x_hat.reshape((n, -1)))

        return {
            f'z_{split_name}': z,
            f'rec_{split_name}': MSE,
            f'rec_time_{split_name}': rec_time,
            f'transform_time_{split_name}': transform_time,
        }


class PHATE(phate.PHATE, BaseModel):
    """Thin wrapper for PHATE to work with torch datasets."""
    def __init__(self, threshold=PROC_THRESHOLD, procrustes_batches_size=1000,
                 procrustes_lm = 1000, **kwargs):
        self.threshold = threshold
        self.procrustes_batches_size = procrustes_batches_size
        self.procrustes_lm = procrustes_lm
        super().__init__(**kwargs)

    # def fit(self, X):
    #     x, _ = X.numpy()
    #     super().fit(x)

    def fit_transform(self, X):
        x, _ = X.numpy()

        if x.shape[0] < self.threshold:
            result = super().fit_transform(x)
        else:
            print('            Fitting procrustes...')
            result = self.fit_transform_procrustes(x)
            # Use procrustes method 
            
            
            pass
        return result
    
    def fit_transform_procrustes(self, x):
        "each batch has procrustes_lm + procrustes_batches_size observations"
        # select procrustes_lm from the dataset (we can use different approaches
        # for now just a random selection from the data, we can shuffle it,
        # before everything)    
        lm_points = x[:self.procrustes_lm, :]
        initial_embedding = super().fit_transform(lm_points)
        result = [initial_embedding]
        remaining_x = x[self.procrustes_lm:, :]
        while len(remaining_x) != 0:
            if len(remaining_x) >= self.procrustes_batches_size:
                new_points = remaining_x[:self.procrustes_batches_size, :]
                remaining_x = np.delete(remaining_x,
                                        np.arange(self.procrustes_batches_size), 
                                        axis = 0)
            else:
                new_points = remaining_x
                remaining_x = np.delete(remaining_x, 
                                        np.arange(len(remaining_x)),
                                        axis = 0)
                
            subsetx = np.vstack((lm_points, new_points))
            subset_embedding = super().fit_transform(subsetx)
            
            d, Z, tform = procrustes(initial_embedding, 
                                     subset_embedding[:self.procrustes_lm,:])
            
            subset_embedding_transformed = np.dot(
                subset_embedding[self.procrustes_lm:, :], 
                        tform['rotation']) + tform['translation']
            
            result.append(subset_embedding_transformed)
        return np.vstack(result)
         
    # def transform(self, X):
    #     x, _ = X.numpy()
    #     return super().transform(x)


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

    def fit(self, X, epochs=None, epoch_offset=0):
        if epochs is None:
            epochs = self.epochs

        # Reproducibility
        torch.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if self.torch_module is None:
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

        for epoch in range(epochs):
            print(f'            Epoch {epoch + epoch_offset}...')
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



class GRAE(AE):
    """Standard GRAE class."""

    def __init__(self, *, lam=100, embedder=PHATE, embedder_args=dict(), relax=True, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.lam_original = lam

        self.embedder = embedder(**embedder_args,
                                 random_state=self.random_state, n_components=self.n_components)
        self.z = None
        self.relax = relax

    def fit(self, X):
        # Find manifold learning embedding
        print('       Fitting embedding...')
        emb = scipy.stats.zscore(self.embedder.fit_transform(X))
        self.z = torch.from_numpy(emb).float().to(device)

        print('       Fitting GRAE...')
        super().fit(X)


    def apply_loss(self, x, x_hat, z, idx):
        """Apply geometric loss and reconstruction loss.

        :param x: Input batch.
        :param x_hat: Reconstructed batch
        :param z:  Embedded batch.
        :param idx: Sample indices
        :return:
        """
        if self.lam > 0:
            loss = self.criterion(x, x_hat) + self.lam * self.criterion(z, self.z[idx])
        else:
            loss = self.criterion(x, x_hat)

        loss.backward()

    def end_epoch(self, epoch):
        """Function called at the end of every training epoch.

        :param epoch: Current epoch number.
        :return:
        """
        if self.relax:
            # Current epoch
            # Update lambda
            self.lam = (-self.epochs*np.exp((epoch - (self.epochs/2))*0.2))/(1+np.exp((epoch - (self.epochs/2))*0.2)) \
                +  self.lam_original


class SmallGRAE(GRAE):
    def __init__(self, *, embedder_args=dict(), threshold=PROC_THRESHOLD, **kwargs):
        super().__init__(lam=.1,
                         embedder=PHATE,
                         embedder_args=embedder_args,
                         threshold=threshold,
                         relax=False,
                         **kwargs)

class LargeGRAE(GRAE):
    def __init__(self, *, embedder_args=dict(), threshold=PROC_THRESHOLD, **kwargs):
        super().__init__(lam=100,
                         embedder=PHATE,
                         embedder_args=embedder_args,
                         threshold=threshold,
                         relax=False,
                         **kwargs)




def procrustes(X, Y, scaling=True, reflection='best'):
    
    """
    Taken from https://stackoverrun.com/es/q/5162566 adaptation of MATLAB to numpy
    """
    
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

