"""Validation and testing experiment functions. Should be called by driver code.

All results are logged to Comet. Comet API key should be in a config file
in the working directory.

"""
import os
import time
import copy

from comet_ml import Experiment

import src.data
import src.models
from src.experiments.utils import save_dict
from src.metrics import score_model


def parse_params(exp_params):
    """Parse experiment parameters.

    Make sure at least model_name, dataset_name and seed are present. Other keys are assumed to be model parameters.
    Extract model, dataset and seed. Return the rest in one dict.

    Args:
        exp_params(dict): Experiment parameters.

    Returns:
        (tuple) tuple containing:
            model_name(str): Model name.
            dataset_name(str): Dataset name.
            random_state(int): Experiment seed.
            model_params(dict): All other keys are assumed to be model parameters.
    """
    if not all(name in exp_params for name in ('model_name', 'dataset_name', 'random_state')):
        raise Exception('Experiment should at least specify model, dataset and seed.')

    # Copy parameters
    exp_params = copy.deepcopy(exp_params)

    # Extract main keys
    model_name = exp_params.pop('model_name')
    dataset_name = exp_params.pop('dataset_name')
    random_state = exp_params.pop('random_state')

    # Convert some values to int
    for key, value in exp_params.items():
        if key in ('epochs', 'batch_size', 't', 'knn', 'n_neighbors', 'subsample', 'patience'):
            exp_params[key] = int(value)

    return model_name, dataset_name, random_state, exp_params


def fit_test(exp_params, data_path, write_path, others=None, custom_tag=None):
    """Fit model and compute metrics on both train and test sets.

    Also log plot and embeddings to comet.

    Args:
        exp_params(dict): Parameter dict. Should at least have keys model_name, dataset_name & random_state. Other
        keys are assumed to be model parameters.
        data_path(str): Data directory.
        write_path(str): Where temp files can be written.
        others(dict): Other things to log to Comet experiment.
        custom_tag(str): Custom tag for Comet experiment.

    """
    # Comet experiment
    exp = Experiment(parse_args=False)
    exp.disable_mp()
    exp.add_tag('test')
    exp.log_parameters(exp_params)

    if others is not None:
        exp.log_others(others)

    if custom_tag is not None:
        exp.add_tag(custom_tag)

    # Parse experiment parameters
    model_name, dataset_name, random_state, model_params = parse_params(exp_params)

    # Fetch and split dataset.
    data_train_full = getattr(src.data, dataset_name)(split='train', random_state=random_state, data_path=data_path)
    data_test = getattr(src.data, dataset_name)(split='test', random_state=random_state, data_path=data_path)
    data_train, data_val = data_train_full.validation_split(fold=8)

    # Model
    m = getattr(src.models, model_name)(random_state=random_state, **model_params)
    m.comet_exp = exp  # Used by DL models to log metrics between epochs
    m.write_path = write_path
    m.data_val = data_val  # For early stopping

    # Benchmark fit time
    fit_start = time.time()

    m.fit(data_train)

    fit_stop = time.time()

    fit_time = fit_stop - fit_start

    # Log plot
    m.plot(data_train, data_val, title=f'{model_name} : {dataset_name}')

    # Score test results first to avoid UMAP bug. See issue #515 of their repo.
    test_z, test_metrics = score_model(dataset_name=dataset_name, model=m, dataset=data_test)

    if model_name == 'UMAP' and dataset_name == 'Embryoid':
        # Reconstructing full training set in reasonable time is very long with UMAP on Embryoid. Skip it.
        train_z, train_metrics = m.transform(data_train), dict()
    else:
        train_z, train_metrics = score_model(dataset_name=dataset_name, model=m, dataset=data_train)

    with exp.train():
        _, train_y = data_train.numpy()

        # Log train metrics
        exp.log_metric(name='fit_time', value=fit_time)
        exp.log_metrics(train_metrics)

    with exp.test():
        _, test_y = data_test.numpy()

        # Log train metrics
        exp.log_metrics(test_metrics)

    # Log embedding as .npy file
    file_name = os.path.join(write_path, f'emb_{model_name}_{dataset_name}.npy')
    save_dict(dict(train_z=train_z,
                   train_y=train_y,
                   test_z=test_z,
                   test_y=test_y,
                   random_state=random_state,
                   dataset_name=dataset_name,
                   model_name=model_name),
              file_name)
    file = open(file_name, 'rb')
    exp.log_asset(file, file_name=file_name)
    file.close()
    os.remove(file_name)

    # Log marker to mark successful experiment
    exp.log_other('success', 1)


def fit_validate(exp_params, k, data_path, write_path, others=None, custom_tag=None):
    """Fit model and compute metrics on train and validation set. Intended for hyperparameter search.

    Only logs metric and scatter plot of final embedding.

    Args:
        exp_params(dict): Parameter dict. Should at least have keys model_name, dataset_name & random_state. Other
        keys are assumed to be model parameters.
        k(int): Fold identifier.
        data_path(str): Data directory.
        write_path(str): Where to write temp files.
        others(dict): Other things to log to Comet experiment.
        custom_tag(str): Custom tag for comet experiment.

    """
    # Comet experiment
    exp = Experiment(parse_args=False)
    exp.disable_mp()
    exp.add_tag('hyper')
    exp.log_parameters(exp_params)

    if others is not None:
        exp.log_others(others)

    if custom_tag is not None:
        exp.add_tag(custom_tag)

    # Parse experiment parameters
    model_name, dataset_name, random_state, model_params = parse_params(exp_params)

    # Fetch and split dataset.
    data_train = getattr(src.data, dataset_name)(split='train', random_state=random_state, data_path=data_path)
    data_train, data_val = data_train.validation_split(k)

    # Model
    m = getattr(src.models, model_name)(random_state=random_state, **model_params)
    m.comet_exp = exp
    m.write_path = write_path
    m.data_val = data_val

    with exp.train():
        m.fit(data_train)

        # Log plot
        m.plot(data_train, data_val, title=f'{model_name} : {dataset_name}')

        # Score val results first to avoid UMAP bug. See issue #515 of their repo.
        val_z, val_metrics = score_model(dataset_name=dataset_name, model=m, dataset=data_val, mse_only=True)

        if model_name != 'UMAP' or dataset_name != 'Embryoid':
            # Do not benchmark train set for UMAP to save time
            train_z, train_metrics = score_model(dataset_name=dataset_name, model=m, dataset=data_train, mse_only=True)

            # Log train metrics
            exp.log_metrics(train_metrics)

    with exp.validate():
        # Log train metrics
        exp.log_metrics(val_metrics)

    # Log marker to mark successful experiment
    exp.log_other('success', 1)
