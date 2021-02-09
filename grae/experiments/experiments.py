"""Validation and testing experiment functions. Should be called by driver code.

All results are logged to Comet. Comet API key should be in a config file
in the working directory.

"""
import os
import time
import copy

from comet_ml import Experiment

import grae.data
import grae.models
from grae.experiments.utils import save_dict
from grae.metrics import EmbeddingProber
from grae.experiments.hyperparameter_config import FOLD_SEEDS


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


def fit_test(exp_params, data_path, k, write_path, others=None, custom_tag=''):
    """Fit model and compute metrics on both train and test sets.

    Also log plot and embeddings to comet.

    Args:
        exp_params(dict): Parameter dict. Should at least have keys model_name, dataset_name & random_state. Other
        keys are assumed to be model parameters.
        k(int): Fold identifier.
        data_path(str): Data directory.
        write_path(str): Where temp files can be written.
        others(dict): Other things to log to Comet experiment.
        custom_tag(str): Custom tag for Comet experiment.

    """
    # Comet experiment
    exp = Experiment(parse_args=False)
    exp.disable_mp()
    custom_tag += '_test'
    exp.add_tag(custom_tag)
    exp.log_parameters(exp_params)

    if others is not None:
        exp.log_others(others)

    # Parse experiment parameters
    model_name, dataset_name, random_state, model_params = parse_params(exp_params)

    # Fetch and split dataset.
    data_train_full = getattr(grae.data, dataset_name)(split='train', random_state=random_state, data_path=data_path)
    data_test = getattr(grae.data, dataset_name)(split='test', random_state=random_state, data_path=data_path)
    data_train, data_val = data_train_full.validation_split(random_state=FOLD_SEEDS[k])

    # Model
    m = getattr(grae.models, model_name)(random_state=FOLD_SEEDS[k], **model_params)
    m.comet_exp = exp  # Used by DL models to log metrics between epochs
    m.write_path = write_path
    m.data_val = data_val  # For early stopping

    # Benchmark fit time
    fit_start = time.time()

    m.fit(data_train)

    fit_stop = time.time()

    fit_time = fit_stop - fit_start

    # Log plots
    m.plot(data_train, data_test, title=f'{model_name}_{dataset_name}')
    if dataset_name in ['Faces', 'RotatedDigits', 'UMIST', 'Tracking', 'COIL100', 'Teapot']:
        m.view_img_rec(data_train, choice='random', title=f'{model_name}_{dataset_name}_train_rec')
        m.view_img_rec(data_test, choice='best', title=f'{model_name}_{dataset_name}_test_rec_best')
        m.view_img_rec(data_test, choice='worst', title=f'{model_name}_{dataset_name}_test_rec_worst')
    elif dataset_name in ['ToroidalHelices', 'Mammoth'] or 'SwissRoll' in dataset_name:
        m.view_surface_rec(data_train, title=f'{model_name}_{dataset_name}_train_rec', dataset_name=dataset_name)
        m.view_surface_rec(data_test, title=f'{model_name}_{dataset_name}_test_rec', dataset_name=dataset_name)

    # Score models
    prober = EmbeddingProber()
    prober.fit(model=m, dataset=data_train_full)

    with exp.train():
        train_z, train_metrics = prober.score(data_train_full)
        _, train_y = data_train.numpy()

        # Log train metrics
        exp.log_metric(name='fit_time', value=fit_time)
        exp.log_metrics(train_metrics)

    with exp.test():
        test_z, test_metrics = prober.score(data_test)
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


def fit_validate(exp_params, k, data_path, write_path, others=None, custom_tag=''):
    """Fit model and compute metrics on train and validation set. Intended for hyperparameter search.

    Only logs final metrics and scatter plot of final embedding.

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
    custom_tag += '_validate'
    exp.add_tag(custom_tag)
    exp.log_parameters(exp_params)

    if others is not None:
        exp.log_others(others)

    # Parse experiment parameters
    model_name, dataset_name, random_state, model_params = parse_params(exp_params)

    # Fetch and split dataset.
    data_train = getattr(grae.data, dataset_name)(split='train', random_state=random_state, data_path=data_path)
    data_train, data_val = data_train.validation_split(random_state=FOLD_SEEDS[k])

    # Model
    m = getattr(grae.models, model_name)(random_state=FOLD_SEEDS[k], **model_params)
    m.write_path = write_path
    m.data_val = data_val

    with exp.train():
        m.fit(data_train)

        # Log plot
        m.comet_exp = exp
        m.plot(data_train, data_val, title=f'{model_name} : {dataset_name}')

        # Probe embedding
        prober = EmbeddingProber()
        prober.fit(model=m, dataset=data_train, mse_only=True)
        train_z, train_metrics = prober.score(data_train, is_train=True)

        # Log train metrics
        exp.log_metrics(train_metrics)

    with exp.validate():
        val_z, val_metrics = prober.score(data_val)

        # Log train metrics
        exp.log_metrics(val_metrics)

    # Log marker to mark successful experiment
    exp.log_other('success', 1)
