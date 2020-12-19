"""Validation and testing experiment functions. Should be called by driver code.

Everytime fit, fit_test and fit_validate are called, one experiment is logged in Comet.

"""
import os
import time

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

    # Extract main keys
    model_name = exp_params.pop('model_name')
    dataset_name = exp_params.pop('dataset_name')
    random_state = exp_params.pop('random_state')

    # Convert some values to int
    for key, value in exp_params.items():
        if key in ('epochs', 'batch_size', 't', 'knn', 'n_neighbors', 'subsample'):
            exp_params[key] = int(value)

    return model_name, dataset_name, random_state, exp_params


# def fit(exp_params):


def fit_test(exp_params, custom_tag=None):
    """Fit model and compute metrics on both train and test sets.

    Also log plot and embeddings to comet.

    Args:
        exp_params(dict): Parameter dict. Should at least have keys model_name, dataset_name & random_state. Other
        keys are assumed to be model parameters.
        custom_tag(str): Custom tag for Comet experiment.

    """
    # Comet experiment
    exp = Experiment()
    exp.disable_mp()
    exp.add_tag('full')
    if custom_tag is not None:
        exp.add_tag(custom_tag)
    exp.log_parameters(exp_params)

    # Parse experiment parameters
    model_name, dataset_name, random_state, model_params = parse_params(exp_params)

    # Fetch and split dataset.
    data_train = getattr(src.data, dataset_name)(split='train', random_state=random_state)
    data_test = getattr(src.data, dataset_name)(split='test', random_state=random_state)

    # Model
    m = getattr(src.models, model_name)(random_state=random_state, **model_params)
    m.set_comet(exp)  # Used by DL models to log metrics between epochs

    # Benchmark fit time
    fit_start = time.time()

    m.fit(data_train)

    fit_stop = time.time()

    fit_time = fit_stop - fit_start

    # Log plot
    m.plot(data_train, data_test, title=dataset_name, comet_exp=exp)

    # Score test results first to avoid UMAP bug. See issue #515 of their repo.
    test_z, test_metrics = score_model(dataset_name=dataset_name, model=m, dataset=data_test)
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
    file_name = f'emb_{model_name}_{dataset_name}.npy'
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


def fit_validate(exp_params, k, custom_tag=None):
    """Fit model and compute metrics on train and validation set. Intended for hyperparameter search.

    Does not log any asset to comet, only metrics.

    Args:
        exp_params(dict): Parameter dict. Should at least have keys model_name, dataset_name & random_state. Other
        keys are assumed to be model parameters.
        k(int): Fold identifier.
        custom_tag(str): Custom tag for comet experiment.

    """
    # Comet experiment
    exp = Experiment()
    exp.disable_mp()
    exp.add_tag('hyper')
    if custom_tag is not None:
        exp.add_tag(custom_tag)
    exp.log_parameters(exp_params)
    exp.log_other('fold', k)

    # Parse experiment parameters
    model_name, dataset_name, random_state, model_params = parse_params(exp_params)

    # Fetch and split dataset.
    data_train = getattr(src.data, dataset_name)(split='train', random_state=random_state)
    data_train, data_val = data_train.validation_split(k)

    # Model
    m = getattr(src.models, model_name)(random_state=random_state, **model_params)

    with exp.train():
        m.fit(data_train)

        # Score val results first to avoid UMAP bug. See issue #515 of their repo.
        val_z, val_metrics = score_model(dataset_name=dataset_name, model=m, dataset=data_val, mse_only=True)
        train_z, train_metrics = score_model(dataset_name=dataset_name, model=m, dataset=data_train, mse_only=True)

        _, train_y = data_train.numpy()

        # Log train metrics
        exp.log_metrics(train_metrics)

    with exp.validate():
        _, test_y = data_val.numpy()

        # Log train metrics
        exp.log_metrics(val_metrics)
