"""Routine to score embeddings produced by the main experiment script."""
import pandas as pd
import os

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

import src.data
from src.experiments.utils import load_dict

# Metrics to compute
METRICS = ['fit_time', 'R2', 'reconstruction']


class Book:
    """Object to save metrics and associated data.

    Add entries with add_entry and return a final dataframe with get_df.
    """

    def __init__(self, datasets, models, metrics):
        """Init.

        Args:
            datasets(List[str]): List of allowed dataset names.
            models(List[str]): List of allowed model names.
            metrics(List[str]): List of allowed metrics
        """
        self.col = ['model', 'dataset', 'run', 'split'] + metrics  # DataFrame columns
        self.log = list()  # List of lists to store entries
        self.models = models
        self.datasets = datasets
        self.splits = ('train', 'test')
        self.metrics = metrics

    def add_entry(self, model, dataset, run, split, **kwargs):
        """Add entry to book.

        Args:
            model(str): Model name.
            dataset(str): Dataset name.
            run(int): Run number.
            split(str): Split name ('train' or 'test').
            **kwargs(Dict[str, float]): Metric values. Key should be the metric name as provided in self.metrics.

        """
        # Proof read entry
        self.check(model, dataset, split, kwargs)

        metrics_ordered = [kwargs[k] for k in self.metrics]

        signature = [model, dataset, run, split]
        entry = signature + metrics_ordered

        if len(entry) != len(self.col):
            raise Exception('Entry size is wrong.')

        self.log.append(entry)

    def check(self, model, dataset, split, kwargs):
        """Check values of arguments.

        Args:
            model(str): Model name.
            dataset(str): Dataset name.
            split(str): Split name ('train' or 'test').
            kwargs(Dict[str, float]): Metric values. Key should be the metric name as provided in self.metrics.

        Raises:
            ValueError : If arguments are not in the lists declared in the init method.

        """
        if model not in self.models:
            raise ValueError('Invalid model name.')

        if dataset not in self.datasets:
            raise ValueError('Invalid dataset name.')

        if split not in self.splits:
            raise ValueError('Invalid split name.')

        if len(kwargs.keys()) != len(self.metrics):
            raise ValueError('Wrong number of metrics.')

        for key in kwargs.keys():
            if key not in self.metrics:
                raise ValueError(f'Trying to add undeclared metric {key}')

    def get_df(self):
        """Return all results accumulated in self.log.

        Returns:
            DataFrame: Results.

        """
        return pd.DataFrame.from_records(self.log, columns=self.col)


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


def score(id_, models, datasets):
    """Score embeddings of an experiment.

    Compute R^2 (see radial_regression and latent_regression) to assess if the embeddings are faithful to the latent
    factors. Compute reconstruction to assess how invertible the embeddings are.

    Save all results in a csv file under ./results/id_

    Args:
        id_(int): ID of the experiment.
        models(List[str]): List of model names.
        datasets(List[str]): List of dataset names.

    """
    path = os.path.join(
        os.path.dirname(__file__),
        os.path.join('..', '..', 'results', id_)
    )
    # File to save data
    file_name = os.path.join(path, 'metrics.csv')

    # Object to keep track of results
    book = Book(models=models,
                datasets=datasets,
                metrics=METRICS)


    # Iterate over all embeddings
    for subdir, dirs, files in os.walk(os.path.join(path, 'embeddings')):

        for file in files:
            dir_list = os.path.normpath(subdir).split(os.sep)
            model, dataset = dir_list[-2:]

            if dataset not in datasets:
                continue
            if model not in models:
                continue

            filepath = subdir + os.sep + file
            print(f'Scoring {filepath}...')
            data = load_dict(filepath)

            n_components = data['z_train'].shape[1]

            # Hot fix to work with the final embeddings despite changes changes to the new code
            if 'rec_train' in data:
                # Compatibility with old format
                data['MSE_train'] = data['rec_train']
                data['MSE_test'] = data['rec_test']

            run_no = int(os.path.splitext(file)[0].split('_')[1])

            if dataset == 'RotatedDigits' and model == 'DiffusionNet' and run_no in [8, 9]:
                # Two runs from the final DiffusionNet dataset failed, do not score them
                continue
            # End of hot fix

            dataset_seed = data['dataset_seed']
            run_seed = data['run_seed']

            # Score embedding
            # Score both splits
            for split in ('train', 'test'):
                metrics = dict()

                rec_key = f'MSE_{split}'

                # Fit linear regressions on a given split
                if n_components == 2:
                    # Only fit a regression of latent factors for 2D embeddings
                    X = getattr(src.data, dataset)(split=split, random_state=dataset_seed)
                    y = X.get_latents()
                    z = data[f'z_{split}']

                    if dataset in ['Teapot', 'RotatedDigits']:
                        # Angle-based regression for circular manifolds
                        r2 = radial_regression(z, *y.T)
                    elif dataset in ['UMIST']:
                        # UMIST has a class-like structure that should be accounted for
                        labels = y[:, 0]
                        angles = y[:, 1:]
                        r2 = latent_regression(z, angles, labels=labels)
                    else:
                        r2 = latent_regression(z, y)

                    metrics.update({'R2': np.mean(r2)})
                else:
                    metrics.update({'R2': None})

                metrics.update({'reconstruction': data[rec_key]})

                fit_time = data['fit_time'] if split == 'train' else np.nan
                metrics.update({'fit_time': fit_time})

                book.add_entry(model=model, dataset=dataset, run=run_seed, split=split, **metrics)

    # Save results
    df = book.get_df()
    df.to_csv(file_name)
