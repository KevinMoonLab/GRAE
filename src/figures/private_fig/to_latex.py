"""Function to generate latex tables."""
import os

import pandas as pd
import numpy as np

from src.figures.name_getters import get_model_name, get_dataset_name, get_metrics_name

# Color Gradient
GRADIENT = ["2F781E", "4D8418", "699010", "879C06", "A5A600", "B6A100", "C69B00",
            "D69400", "D87D00", "D96300", "D94300", "D70b0B"]


def to_latex(id, split, model_list, dataset_list, digits=4):
    path = os.path.join(
        os.path.dirname(__file__),
        os.path.join('..', '..', 'results', id)
    )
    file_path = os.path.join(path, 'metrics.csv')
    target_path = os.path.join(path, f'table_{split}.txt')

    # g.reverse()
    gradient = np.array(GRADIENT)

    # Create smaller interpolated version
    n = len(gradient) - 1
    max_rank = len(model_list)
    interpolate = np.linspace(start=0, stop=n, num=max_rank).astype(int)
    gradient = gradient[interpolate[np.arange(len(model_list))]]

    df = pd.read_csv(file_path, index_col=0)

    df = df[df['model'].isin(model_list)]
    df = df[df['dataset'].isin(dataset_list)]

    # Keep only test data and drop useless columns
    df = df[df['split'] == split]

    df = df.drop(columns=['split', 'run'])

    # for base in ('pearson', 'spearman', 'mutual_information'):
    #     df[f'{base}'] = (df[f'{base}_source_1'] + df[f'{base}_source_2']) / 2
    #     df[f'{base}_ICA'] = (df[f'{base}_ICA_source_1'] + df[f'{base}_ICA_source_2']) / 2
    #     # df[f'{base}_slice'] = (df[f'{base}_slice_source_1'] + df[f'{base}_slice_source_2']) / 2

    # Keep only relevant columns
    # df = df[['dataset', 'model', 'dist_corr',
    #          'pearson',
    #          'pearson_ICA',
    #          'spearman',
    #          'spearman_ICA',
    #          'reconstruction'
    #          ]]

    metrics = ['R2', 'reconstruction']

    if split == 'train':
        keep = ['fit_time'] + metrics
    else:
        keep = metrics

    df = df[['dataset', 'model'] + keep]

    # Provide order for models and datasets if needed
    df['model'] = pd.Categorical(df['model'], model_list)
    df['dataset'] = pd.Categorical(df['dataset'], dataset_list)

    # Mean or sd
    mean = df.groupby(['dataset', 'model']).mean()
    mean = mean.reset_index(level=[0, 1])
    mean = mean.round(digits)

    if 'AE' in model_list:
        # Add relative comparison to AE reconstruction
        mean['rel_reconstruction'] = mean['reconstruction']

        for ds in dataset_list:
            mask = (mean['dataset'] == ds) & (mean['model'] == 'AE')
            quotient = mean.loc[mask, 'reconstruction'].iloc[0]
            mean.loc[mean['dataset'] == ds, 'rel_reconstruction'] /= quotient

        mean['rel_reconstruction'] -= 1
        mean['rel_reconstruction'] *= 100
        mean['rel_reconstruction'] = mean['rel_reconstruction'].round(1)

    if split == 'train':
        mean['fit_time'] /= 60
        mean['fit_time'] = mean['fit_time'].round(2)

    # Prettify column names
    mean['model'] = mean['model'].map(get_model_name)
    mean['dataset'] = mean['dataset'].map(get_dataset_name)

    # Build rank dataframe
    # 2 means bold. 1 means bold + underline.
    mean.set_index(['dataset', 'model'], inplace=True)

    df_rank = pd.DataFrame()

    for m in list(mean):
        # Higher is better
        ascending = False

        if m == 'reconstruction' or m == 'rel_reconstruction' or m == 'fit_time':
            # Lower is better
            ascending = True

        rank = mean.groupby(level=[0])[m].rank(method='min', ascending=ascending)
        df_rank[m] = rank

    mean.loc[:, metrics] = mean.loc[:, metrics].applymap(lambda x: ("{:." + str(digits) + "f}").format(x))


    # Add percentage to relative reconstruction
    if 'AE' in model_list:
        mean.loc[:, ['rel_reconstruction']] = mean.loc[:, ['rel_reconstruction']].applymap(
            lambda x: ("{:." + str(1) + "f}").format(x))
        mean['rel_reconstruction'] += ' \%'

    if split == 'train':
        mean.loc[:, ['fit_time']] = mean.loc[:, ['fit_time']].applymap(
            lambda x: ("{:." + str(2) + "f}").format(x))

    # Prettify column names
    mean = mean.rename(columns=get_metrics_name)

    # Generate colored table
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if not np.isnan(df_rank.iloc[i, j]):
                rank = int(df_rank.iloc[i, j])
            else:
                rank = len(model_list)

            # Colors and ranks
            mean.iloc[i, j] += f' ({rank})'

            if rank == 1.:
                mean.iloc[i, j] = r'{\ul \textbf{' + mean.iloc[i, j] + '}}'
            elif rank == 2.:
                mean.iloc[i, j] = r'\textbf{' + mean.iloc[i, j] + '}'

            mean.iloc[i, j] = '\color[HTML]{' + gradient[rank - 1] + '}' + mean.iloc[i, j]

    result = f'% {split} split\n' + (
                r'\resizebox{\textwidth}{!}{' + mean.to_latex(escape=False, multirow=True) + '}').replace('\cline{1-9}',
                                                                                                          '\hline\hline').replace(
        'nan', 'n/a').replace('\cline{1-5}', '\hline')

    with open(target_path, "w") as text_file:
        text_file.write(result)