"""Parent class for all project models."""
import time

import matplotlib
from sklearn.metrics import mean_squared_error


class BaseModel:
    """All models should subclass BaseModel."""

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

        if title is not None:
            plt.title(title, fontsize=20)
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
            self.comet_exp.log_figure(figure=plt)
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

    # def rec_plot(self, x, im_shape, N=8):
    #     np.random.seed(self.random_state)
    #
    #     x_hat = self.reconstruct(x)
    #     x, _ = x.numpy()
    #
    #     sample_mask = np.random.choice(x.shape[0], size=N, replace=False)
    #     x = x[sample_mask]
    #     x_hat = x_hat[sample_mask]
    #
    #     fig, ax = plt.subplots(N, 2, figsize=(2 * 3.5, N * 3.5))
    #
    #     for i in range(ax.shape[0]):
    #         original = x[i].reshape(im_shape)
    #         reconstructed = x_hat[i].reshape(im_shape)
    #
    #         original = ndimage.rotate(original, -90, reshape=False)
    #         reconstructed = ndimage.rotate(reconstructed, -90, reshape=False)
    #
    #         for j, im in enumerate((original, reconstructed)):
    #             axis = ax[i, j]
    #             axis.imshow(im, cmap='Greys_r')
    #             axis.set_xticks([])
    #             axis.set_yticks([])
    #
    #     plt.show()
