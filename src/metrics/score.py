"""Routine to score embeddings."""
import pandas as pd
import os

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

import src.data
from src.metrics.dis_metrics import dis_metrics, dis_metrics_1D
from src.figures.utils import load_dict


class Book():
    """Class log metrics."""

    def __init__(self, datasets, models, metrics):
        self.col = ['model', 'dataset', 'run', 'split'] + metrics
        self.log = list()
        self.models = models
        self.datasets = datasets
        self.splits = ('train', 'test')
        self.metrics = metrics

    def add_entry(self, model, dataset, run, split, **kwargs):
        # Proof read entry
        self.check(model, dataset, split)
        self.check_metrics(kwargs)

        metrics_ordered = [kwargs[k] for k in self.metrics]

        signature = [model, dataset, run, split]
        entry = signature + metrics_ordered

        if len(entry) != len(self.col):
            raise Exception('Entry size is wrong.')

        self.log.append(entry)

    def check(self, model, dataset, split):
        if model not in self.models:
            raise Exception('Invalid model name.')

        if dataset not in self.datasets:
            raise Exception('Invalid dataset name.')

        if split not in self.splits:
            raise Exception('Invalid split name.')

    def check_metrics(self, kwargs):
        if len(kwargs.keys()) != len(self.metrics):
            raise Exception('Wrong number of metrics.')

        for key in kwargs.keys():
            if key not in self.metrics:
                raise Exception(f'Trying to add undeclared metric {key}')

    def get_df(self):
        return pd.DataFrame.from_records(self.log, columns=self.col)


def refine_df(df, df_metrics):
    df_group = df.groupby(['split', 'dataset', 'model'])
    mean = df_group.mean().drop(columns=['run']).round(4)

    # Add rank columns
    for m in df_metrics:
        # Higher is better
        ascending = False

        if m == 'reconstruction' or m.split('_')[0] == 'mrre':
            # Lower is better
            ascending = True

        loc = mean.columns.get_loc(m) + 1
        rank = mean.groupby(['split', 'dataset'])[m].rank(method='min', ascending=ascending)
        mean.insert(loc=loc, column=f'{m}_rank', value=rank)

    return mean


# Define required metrics
# 2D Disentanglement metrics
# DIS_METRICS = ['dist_corr',
#                'pearson_source_1', 'pearson_source_2',
#                'pearson_ICA_source_1', 'pearson_ICA_source_2',
#                # 'pearson_slice_source_1', 'pearson_slice_source_2',
#                'spearman_source_1', 'spearman_source_2',
#                'spearman_ICA_source_1', 'spearman_ICA_source_2',
#                # 'spearman_slice_source_1', 'spearman_slice_source_2',
#                'mutual_information_source_1', 'mutual_information_source_2',
#                'mutual_information_ICA_source_1', 'mutual_information_ICA_source_2',
#                # 'mutual_information_slice_source_1', 'mutual_information_slice_source_2'
# ]
DIS_METRICS = ['R2', 'reconstruction']


def radial_regression(cartesian_emb, labels, angles):
    """Regression of the angles of an embedding with an angle ground truth and return R^2.
    Used for datasets such as Teapot and Rotated Digits.

    If multiple classes (rings) are present, will return the R^2 average."""

    if cartesian_emb.shape[1] != 2:
        raise Exception('Radial regression requires conversion to polar coordinates. Will only work'
                        'with 2 dimensions.')

    c = np.unique(labels)
    r2 = list()

    for i in c:
        mask = labels == i
        emb = cartesian_emb[mask]
        centered_emb = emb - emb.mean(axis=0)

        class_angles = angles[mask]
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

        m = Lasso(alpha=.002, fit_intercept=True)
        m.fit(phi, class_angles)
        r2.append(m.score(phi, class_angles))

    return np.mean(r2)

def latent_regression(z, y, labels=None):
    """Regression of latent ground truth factors (y) using embedding z."""
    r2 = list()

    if labels is None:
        labels = np.ones(z.shape[0])

    c = np.unique(labels)

    for i in c :
        mask = labels == i
        z_c = z[mask]
        y_c = y[mask]

        z_scaler = StandardScaler(with_std=True)
        y_scaler = StandardScaler(with_std=True)

        z_c = z_scaler.fit_transform(z_c)
        y_c = y_scaler.fit_transform(y_c)

        for latent in y_c.T:
            m = Lasso(alpha=.002, fit_intercept=False)
            m.fit(z_c, latent)
            r2.append(m.score(z_c, latent))

    return np.mean(r2)


def score(id, model_list, dataset_list):
    path = os.path.join(
        os.path.dirname(__file__),
        os.path.join('..', '..', 'results', id)
    )
    # File to save data
    file_name = os.path.join(path, 'metrics.csv')

    # Loogers for results
    book = Book(models=model_list,
                datasets=dataset_list,
                metrics=DIS_METRICS)

    # Iterate over all embeddings
    for subdir, dirs, files in os.walk(os.path.join(path, 'embeddings')):

        for file in files:
            dir_list = os.path.normpath(subdir).split(os.sep)
            model, dataset = dir_list[-2:]

            if dataset not in dataset_list:
                continue
            if model not in model_list:
                continue

            filepath = subdir + os.sep + file
            print(f'Scoring {filepath}...')
            data = load_dict(filepath)

            dataset_seed = data['dataset_seed']
            run_seed = data['run_seed']

            # Score embedding

            # Score both splits
            for split in ('train', 'test'):
                metrics = dict()

                # Fit linear regressions on train split
                X = getattr(src.data, dataset)(split=split, seed=dataset_seed)
                y = X.get_latents()
                z = data[f'z_{split}']

                if dataset in ['Teapot', 'RotatedDigits', 'COIL100']:
                    r2 = radial_regression(z, *y.T)
                elif dataset in ['UMIST']:
                    labels, angles = y.T
                    r2 = latent_regression(z, angles, labels=labels)
                else:
                    r2 = latent_regression(z, y)

                # X = getattr(src.data, dataset)(split=split, seed=dataset_seed)
                # y_1, y_2 = X.get_source()  # Fetch ground truth

                # if dataset == 'SwissRoll':
                #     # The 'Slice' on Swiss Roll is out of distribution with different mean and variance.
                #     # Rescale data based on test mean to avoid bias in the predictions.
                #     z = z_scaler.fit_transform(data[f'z_{split}'])
                #     y_1, y_2 = y_scaler.fit_transform(np.vstack((y_1, y_2)).T).T
                # else:
                #     z = z_scaler.transform(data[f'z_{split}'])
                #     y_1, y_2 = y_scaler.transform(np.vstack((y_1, y_2)).T).T
                #
                rec_key = f'rec_{split}'

                metrics.update({'R2': np.mean(r2)})
                metrics.update({'reconstruction': data[rec_key]})

                book.add_entry(model=model, dataset=dataset, run=run_seed, split=split, **metrics)

            # for split in ('train', 'test'):
            #     z = data[f'z_{split}']
            #
            #     X = getattr(src.data, dataset)(split='train', seed=dataset_seed)
            #     y_1, y_2 = X.get_source()  # Fetch ground truth
            #
            #
            #     if y_1 is not None and y_2 is not None:
            #         # Compute metrics if dataset has 2D Ground truth
            #         metrics = dis_metrics(y_1, y_2, *z.T)
            #     elif y_1 is not None and y_2 is None:
            #         # Compute metrics if dataset has 1D Ground truth
            #         metrics = dis_metrics_1D(y_1, *z.T)
            #     else:
            #         # Dummy dict if no ground truth
            #         metrics = dict(zip(DIS_METRICS, [None] * len(DIS_METRICS)))
            #
            #     rec_key = f'rec_{split}'
            #
            #     metrics.update({'reconstruction': data[rec_key]})
            #
            #     book.add_entry(model=model, dataset=dataset, run=run_seed, split=split, **metrics)

    # Save results
    df = book.get_df()
    df.to_csv(file_name)
