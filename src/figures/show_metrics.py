"""Function to display average metrics over all runs."""
import os
import copy

import pandas as pd

from src.figures.name_getters import get_model_name, get_dataset_name, get_metrics_name


# Show metrics function
def show_metrics(id_, split, model_list, dataset_list):
    """Average metrics over all runs and return resulting dataframe.

    Args:
        id_(str): ID of the desired experiment, as saved under ./results.
        split (str): 'train' or 'test' to show the relevant metrics.
        model_list(List[str]): Models to plot.
        dataset_list(List[str]): Datasets to plot.

    Returns:
        DataFrame : average metrics by model and by dataset.

    """
    path = os.path.join(
        os.path.dirname(__file__),
        os.path.join('..', '..', 'results', id_)
    )

    file_path = os.path.join(path, 'metrics.csv')

    df = pd.read_csv(file_path, index_col=0)

    df = df[df['model'].isin(model_list)]
    df = df[df['dataset'].isin(dataset_list)]
    df = df[df['split'] == split]

    df = df.drop(columns=['split', 'run', 'fit_time'])

    metrics_datasets = copy.deepcopy(dataset_list)

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
