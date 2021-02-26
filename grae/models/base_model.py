"""Parent class for all project models."""
import time

import matplotlib
import numpy as np
from sklearn.metrics import mean_squared_error

SEED = 42

class BaseModel:
    """All models should subclass BaseModel."""
    def __init__(self):
        """Init."""
        self.comet_exp = None

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        raise NotImplementedError()

    def fit_transform(self, x):
        """Fit model and transform data.

        If model is a dimensionality reduction method, such as an Autoencoder, this should return the embedding of X.

        Args:
            x(BaseDataset): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x.

        """
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        """Transform data.

        If model is a dimensionality reduction method, such as an Autoencoder, this should return the embedding of x.

        Args:
            X(BaseDataset): Dataset to transform.
        Returns:
            ndarray: Embedding of X.

        """
        raise NotImplementedError()

    def inverse_transform(self, x):
        """Take coordinates in the embedding space and invert them to the data space.

        Args:
            x(ndarray): Points in the embedded space with samples on the first axis.
        Returns:
            ndarray: Inverse (reconstruction) of x.

        """
        raise NotImplementedError()

    def fit_plot(self, x_train, x_test=None, cmap='jet', s=15, title=None):
        """Fit x_train and show a 2D scatter plot of x_train (and possibly x_test).

        If x_test is provided, x_train points will be smaller and grayscale and x_test points will be colored.

        Args:
            x_train(BaseDataset): Data to fit and plot.
            x_test(BaseDatasset): Data to plot. Set to None to only plot x_train.
            cmap(str): Matplotlib colormap.
            s(float): Scatter plot marker size.
            title(str): Figure title. Set to None for no title.

        """
        self.plot(x_train, x_test, cmap, s, title, fit=True)

    def plot(self, x_train, x_test=None, cmap='jet', s=15, title=None, fit=False):
        """Plot x_train (and possibly x_test) and show a 2D scatter plot of x_train (and possibly x_test).

        If x_test is provided, x_train points will be smaller and grayscale and x_test points will be colored.
        Will log figure to comet if Experiment object is provided. Otherwise, plt.show() is called.

        Args:
            x_train(BaseDataset): Data to fit and plot.
            x_test(BaseDatasset): Data to plot. Set to None to only plot x_train.
            cmap(str): Matplotlib colormap.
            s(float): Scatter plot marker size.
            title(str): Figure title. Set to None for no title.
            fit(bool): Whether model should be trained on x_train.

        """
        if self.comet_exp is not None:
            # If comet_exp is set, use different backend to avoid display errors on clusters
            matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
        import matplotlib.pyplot as plt

        if not fit:
            z_train = self.transform(x_train)
        else:
            z_train = self.fit_transform(x_train)

        y_train = x_train.targets.numpy()

        if z_train.shape[1] != 2:
            raise Exception('Can only plot 2D embeddings.')

        plt.figure(figsize=(3.5, 3.5))

        if title is not None:
            plt.title(title, fontsize=12)
            plt.xticks([])
            plt.yticks([])

        if x_test is None:
            plt.scatter(*z_train.T, c=y_train, cmap=cmap, s=s)
        else:
            # Train data is grayscale and Test data is colored
            z_test = self.transform(x_test)
            y_test = x_test.targets.numpy()
            plt.scatter(*z_train.T, c='grey', s=s / 10, alpha=.2)
            plt.scatter(*z_test.T, c=y_test, cmap=cmap, s=s)

        if self.comet_exp is not None:
            self.comet_exp.log_figure(figure=plt, figure_name=title)
            plt.clf()
        else:
            plt.show()

    def reconstruct(self, x):
        """Transform and inverse x.

        Args:
            x(BaseDataset): Data to transform and reconstruct.

        Returns:
            ndarray: Reconstructions of x.

        """
        return self.inverse_transform(self.transform(x))

    def score(self, x):
        """Compute embedding of x, MSE on x and performance time of transform and inverse transform on x.

        Args:
            x(BaseDataset): Dataset to score.

        Returns:
            (tuple) tuple containing:
                z(ndarray): Data embedding.
                metrics(dict[float]):
                    MSE(float): Reconstruction MSE error.
                    rec_time(float): Reconstruction time in seconds.
                    transform_time(float): Transform time in seconds.
        """
        n = len(x)

        start = time.time()
        z = self.transform(x)
        stop = time.time()

        transform_time = stop - start

        start = time.time()
        x_hat = self.inverse_transform(z)
        stop = time.time()

        rec_time = stop - start

        x, _ = x.numpy()
        MSE = mean_squared_error(x.reshape((n, -1)), x_hat.reshape((n, -1)))

        return z, {
            'MSE': MSE,
            'transform_time': transform_time,
            'rec_time': rec_time,
        }

    def view_img_rec(self, x, n=8, random_state=42, title=None, choice='random'):
        """View n original images and their reconstructions.

        Only call this method on images dataset. x is expected to be 4D.
        Will show figure or log it to Comet if self.comet_exp was set.

        Args:
            x(BaseDataset): Dataset to sample from.
            n(int): Number of images to sample.
            random_state(int): Seed for sampling.
            title(str): Figure title.
            choice(str): 'random' for n random images in dataset. 'best' for images with best reconstructions. 'worst'
            for images with worst reconstructions.

        """
        if self.comet_exp is not None:
            # If comet_exp is set, use different backend to avoid display errors on clusters
            matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
        import matplotlib.pyplot as plt

        np.random.seed(random_state)

        x_hat = self.reconstruct(x)
        x, _ = x.numpy()

        if choice == 'random':
            sample_mask = np.random.choice(x.shape[0], size=n, replace=False)
        else:
            n_sample = x.shape[0]
            mse = mean_squared_error(x.reshape((n_sample, -1)).T, x_hat.reshape((n_sample, -1)).T, multioutput='raw_values')
            mse_rank = np.argsort(mse)
            if choice == 'worst':
                sample_mask = mse_rank[-n:]
            elif choice == 'best':
                sample_mask = mse_rank[:n]
            else:
                raise Exception('Choice name should be random, best or worst.')

        x_hat = x_hat[sample_mask]
        x = x[sample_mask].reshape(x_hat.shape)

        if x_hat.shape[1] == 1:
            grayscale = True
            x_hat = np.squeeze(x_hat)
            x = np.squeeze(x)
        elif x_hat.shape[1] == 3:
            grayscale = False
            x_hat = np.transpose(x_hat, (0, 2, 3, 1))
            x = np.transpose(x, (0, 2, 3, 1))
        else:
            raise Exception('Invalid number of channels.')

        fig, ax = plt.subplots(2, n, figsize=(n * 3.5, 2.2 + 2 * 3.5))

        for i in range(ax.shape[1]):
            original = x[i]
            reconstructed = x_hat[i]

            for j, im in enumerate((original, reconstructed)):
                axis = ax[j, i]
                if grayscale:
                    axis.imshow(im, cmap='Greys_r')
                else:
                    axis.imshow(im)

                axis.set_xticks([])
                axis.set_yticks([])

        if title is not None:
            fig.suptitle(title, fontsize=40)
        fig.tight_layout()

        if self.comet_exp is not None:
            self.comet_exp.log_figure(figure=plt, figure_name=title)
            plt.clf()
        else:
            plt.show()

    def view_surface_rec(self, x, n_max=1000, random_state=42, title=None, dataset_name=None):
        """View 3D original surface and reconstruction.

        Only call this method on 3D surface datasets. x is expected to be 2D.
        Will show figure or log it to Comet if self.comet_exp was set.

        Args:
            x(BaseDataset): Dataset to sample from.
            n_max(int): Number of points to display.
            random_state(int): Seed for sampling.
            title(str): Figure title.
            dataset_name(str): Dataset name to set customized tilt and rotations.

        """
        if self.comet_exp is not None:
            # If comet_exp is set, use different backend to avoid display errors on clusters
            matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
        import matplotlib.pyplot as plt
        from grae.data.manifolds import set_axes_equal

        np.random.seed(random_state)

        x_hat = self.reconstruct(x)
        x, y = x.numpy()

        if x.shape[0] > n_max:
            sample_mask = np.random.choice(x.shape[0], size=n_max, replace=False)
            x_hat = x_hat[sample_mask]
            x = x[sample_mask]
            y = y[sample_mask]

        scene_dict = dict(SwissRoll=(0, 0), Mammoth=(-15, 90), ToroidalHelices=(30, 0))
        if dataset_name in scene_dict:
            tilt, rotation = scene_dict[dataset_name]
        else:
            tilt, rotation = 0, 0

        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.view_init(tilt, rotation)
        ax.set_title('Input')
        ax.scatter(*x.T, c=y, cmap='jet', edgecolor='k')
        set_axes_equal(ax)

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        ax.view_init(tilt, rotation)
        ax.set_title('Reconstruction')
        ax.scatter(*x_hat.T, c=y, cmap='jet', edgecolor='k')
        set_axes_equal(ax)


        if title is not None:
            fig.suptitle(title, fontsize=20)

        if self.comet_exp is not None:
            self.comet_exp.log_figure(figure=plt, figure_name=title)
            plt.clf()
        else:
            plt.show()

