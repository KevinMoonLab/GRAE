"""Function to display metrics."""
import os
import copy

import pandas as pd

from src.figures.name_maps import get_model_name, get_dataset_name, get_metrics_name


# Show metrics function
def show_metrics(id, split, model_list, dataset_list):
    path = os.path.join(
        os.path.dirname(__file__),
        os.path.join('..', '..', 'results', id)
    )

    file_path = os.path.join(path, 'metrics.csv')

    df = pd.read_csv(file_path, index_col=0)

    df = df[df['model'].isin(model_list)]
    df = df[df['dataset'].isin(dataset_list)]
    df = df[df['split'] == split]

    df = df.drop(columns=['split', 'run'])

    metrics_datasets = copy.deepcopy(dataset_list)

    # Average pearson, spearman and MI over source 1 and source 2
    for base in ('pearson', 'spearman', 'mutual_information'):
        df[f'{base}'] = (df[f'{base}_source_1'] + df[f'{base}_source_2']) / 2
        df[f'{base}_ICA'] = (df[f'{base}_ICA_source_1'] + df[f'{base}_ICA_source_2']) / 2
        # df[f'{base}_slice'] = (df[f'{base}_slice_source_1'] + df[f'{base}_slice_source_2']) / 2

    # Keep only relevant columns
    df = df[['dataset', 'model', 'dist_corr',
             'pearson', 'pearson_ICA',
             # 'pearson_slice',
             'spearman', 'spearman_ICA',
             # 'spearman_slice',
             'mutual_information', 'mutual_information_ICA',
             # 'mutual_information_slice',
             'reconstruction']]

    # Provide order for models and datasets if needed
    df['model'] = pd.Categorical(df['model'], model_list)
    df['dataset'] = pd.Categorical(df['dataset'], metrics_datasets)

    # Prettify names
    df['model'] = df['model'].map(get_model_name)
    df['dataset'] = df['dataset'].map(get_dataset_name)

    # Prettify column names
    df = df.rename(columns=get_metrics_name)

    # Mean
    return df.groupby(['Dataset', 'Model']).mean().round(3)
