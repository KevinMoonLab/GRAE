"""Routine to score embeddings."""
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Metrics to compute
METRICS = ['fit_time', 'R2', 'reconstruction']


def radial_regression(cartesian_emb, labels, angles):
    """Regression of the angles of an embedding with a "polar" ground truth and return R^2.

    Used for datasets such as Teapot and Rotated Digits.
    First center embeddings and use one point to align them, otherwise a rotation may
    break the linear relationship. Compute the R^2 score based on the embedding angles.
    If multiple classes (rings) are present (e.g. Rotated Digits), will return the R^2 average over
    all rings.

    Args:
        cartesian_emb(ndarray): Embedding.
        labels(ndarray): Ground truth classes.
        angles(ndarray): Ground truth angles.

    Returns:
        float: R^2 (or average thereof) of the regressor on the embedding angles.
    """

    if cartesian_emb.shape[1] != 2:
        raise ValueError('Radial regression requires conversion to polar coordinates. Will only work'
                         'with 2 dimensions.')

    c = np.unique(labels)
    r2 = list()

    for i in c:
        mask = labels == i
        emb = cartesian_emb[mask]
        centered_emb = emb - emb.mean(axis=0)

        # Ground truth in class
        class_angles = angles[mask]

        # Polar coordinates of the embedding (we're only interested in the angle here)
        phi = np.arctan2(*centered_emb.T) + np.pi

        # Align smallest angle to prevent arbitrary rotation from affecting the regression
        arg_min = min(class_angles.argmin(), phi.argmin())
        phi -= phi[arg_min]
        phi %= 2 * np.pi
        class_angles -= class_angles[arg_min]
        class_angles %= 2 * np.pi

        corr, _ = pearsonr(phi, class_angles)

        if corr < 0:
            # If the correlation is inversed (negative slope) the minimum angle should be mapped to
            # 2pi otherwise it'll be an outlier in the regression
            phi[arg_min] = 2 * np.pi

        phi = phi.reshape((-1, 1))

        # Compute regression and R^2 score
        m = Lasso(alpha=.002, fit_intercept=True)
        m.fit(phi, class_angles)
        r2.append(m.score(phi, class_angles))

    return np.mean(r2)


def latent_regression(z, y, labels=None):
    """Regression of latent ground truth factors (y) using embedding z.

    Compute a linear regression to predict a ground truth factor based on the embedding coordinates and return the R^2
    score. If more than one ground truth variable is present, fit multiple regressors and return the R^2.

    If sample classes are provided in the labels argument, repeat the above procedure independantly for all classes and
    return the average score.

    Args:
        z(ndarray): Embedding.
        y(ndarray): Ground truth variables, as columns.
        labels(ndarray): Class indices, if embedding needs to be partionned.

    Returns:
        float : R^2 (or average thereof over all classes and ground truth variables) of a linear regressor over the
        embedding coordinates.
    """
    r2 = list()

    # If no class is provided, use a dummy constant class for all samples
    if labels is None:
        labels = np.ones(z.shape[0])

    c = np.unique(labels)

    for i in c:
        mask = labels == i
        z_c = z[mask]
        y_c = y[mask]

        # Rescale data
        z_scaler = StandardScaler(with_std=True)
        y_scaler = StandardScaler(with_std=True)

        z_c = z_scaler.fit_transform(z_c)
        y_c = y_scaler.fit_transform(y_c)

        for latent in y_c.T:
            m = Lasso(alpha=.002, fit_intercept=False)
            m.fit(z_c, latent)
            r2.append(m.score(z_c, latent))

    return np.mean(r2)


def score_model(dataset_name, model, dataset, mse_only=False):
    """Compute embedding of dataset with model. Return embedding and some performance metrics.

    Args:
        dataset_name(str): Name of dataset.
        model(BaseModel): Fitted model.
        dataset(BaseDataset): Dataset to embed and score.
        mse_only(bool): Set to False to compute only MSE. Useful for lightweight computations during
        hyperparameter search.

    Returns:
        (tuple) tuple containing:
            z(ndarray): Data embedding.
            metrics(dict[float]): Dict of metrics.

    """
    metrics = dict()

    # Compute embedding and MSE
    z, rec_metrics = model.score(dataset)
    metrics.update(rec_metrics)

    n_components = z.shape[1]

    # Fit linear regressions on a given split
    if n_components == 2 and not mse_only:
        # Only fit a regression of latent factors for 2D embeddings
        y = dataset.get_latents()

        if dataset_name in ['Teapot', 'RotatedDigits', 'ToroidalHelices', 'COIL100']:
            # Angle-based regression for circular manifolds
            r2 = radial_regression(z, *y.T)
        elif dataset_name in ['UMIST', 'ArtificialTree']:
            # Some datasets have a cluster structure that should be accounted for
            labels = y[:, 0]
            target = y[:, 1:]
            r2 = latent_regression(z, target, labels=labels)
        elif dataset_name in ['Mammoth']:
            # Mammoth dataset does not have a continuous target variable
            r2 = 0
        else:
            r2 = latent_regression(z, y)

        metrics.update({'R2': np.mean(r2)})

    # Add classification accuracy for some problems
    if not mse_only and dataset_name in ['RotatedDigits', 'UMIST', 'ToroidalHelices', 'COIL100', 'ArtificialTree']:
        _, y = dataset.numpy()

        m = LogisticRegression(max_iter=1000)

        m.fit(z, y)

        metrics.update({'Acc': m.score(z, y)})

    return z, metrics
