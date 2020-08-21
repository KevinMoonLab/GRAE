"""Main experiment script.

   All results, including embeddings, plots and latex tables will be saved under results/ID in the project root.
   ID is generated based on current time.
"""
import os
import datetime
import time
import copy

from sklearn.metrics import mean_squared_error
from six.moves import cPickle as pickle  # for performance

import src.data
import src.models
from src.metrics import score
from src.figures import grid_plot, show_metrics, to_latex
from src.experiments.model_params import DEFAULTS, DATASET_PARAMS


# Fit models
# Models and Datasets for experiment
# Names should be the same as the class names defined in the models and datasets modules. GRAE variants can be suffixed
# (ex: GRAE_10) to choose the lambda value
# Specific model arguments can be changed in the model_params.py module
MODELS = ['AE', 'GRAE_100', 'SoftGRAE_100']
DATASETS = ['Teapot', 'Tracking', 'RotatedDigits']

RUNS = 1
RANDOM_STATES = [36087, 63286, 52270, 10387, 40556, 52487, 26512, 28571, 33380,
                 9369, 28478, 4624, 29114, 41915, 6467, 4216, 16025, 34823,
                 29854, 23853]  # 20 random states. Add more if needed.


# Util to save dicts
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

# Create PATH variable and directory to save embeddings
ID = str(int((datetime.datetime.now() - datetime.datetime(2020, 8, 1)).total_seconds() * 1e6))  # Create ID with time
PATH = os.path.join(
    os.path.dirname(__file__),
    os.path.join('..', '..', 'results', ID, 'embeddings')
)

if not os.path.exists(PATH):
    os.makedirs(PATH)
else:
    raise Exception(f'Embedding folder with name {ID} already exists!')

# Experiment loop
for model in MODELS:
    print(f'Training {model}...')

    os.mkdir(os.path.join(PATH, model))

    for i, dataset in enumerate(DATASETS):
        target = os.path.join(PATH, model, dataset)

        os.mkdir(target)

        print(f'   On {dataset}...')
        # Training loop
        for j in range(RUNS):
            print(f'       Run {j + 1}...')

            # Fetch and split dataset. Handle numpy input for some models
            data_train = getattr(src.data, dataset)(split='train', seed=RANDOM_STATES[j])
            data_test = getattr(src.data, dataset)(split='test', seed=RANDOM_STATES[j])

            data_train_np, y_train = data_train.numpy()
            data_test_np, y_test = data_test.numpy()

            # Parse model name. Underscore allows to pass a lambda argument to GRAE variants (ex : GRAE_10)
            arg_list = model.split('_')
            model_name = arg_list[0]
            params = copy.deepcopy(DEFAULTS[model_name])

            if dataset in DATASET_PARAMS[model_name]:
                params.update(DATASET_PARAMS[model_name][dataset])

            # Get model. Call it with lambda if lambda underscore value is passed.
            if len(arg_list) >= 2:
                lam = int(arg_list[1])
                params.update(dict(lam=lam))

            m = getattr(src.models, model_name)(random_state=RANDOM_STATES[j], **params)

            # Benchmark fit time
            fit_start = time.time()

            z_train = m.fit_transform(data_train)

            fit_stop = time.time()

            fit_time = fit_stop - fit_start

            # Benchmark transform time if required
            transform_start = time.time()
            z_test = m.transform(data_test)
            transform_stop = time.time()

            transform_time = transform_stop - transform_start

            inv_train = m.inverse_transform(z_train)
            inv_test = m.inverse_transform(z_test)

            # Compute reconstruction right away to avoid needing to save all reconstructed samples
            rec_train = mean_squared_error(data_train_np, inv_train)
            rec_test = mean_squared_error(data_test_np, inv_test)

            # Save embeddings
            obj = dict(z_train=z_train, z_test=z_test,
                       rec_train=rec_train, rec_test=rec_test,
                       fit_time=fit_time, transform_time=transform_time,
                       dataset_seed=RANDOM_STATES[j], run_seed=RANDOM_STATES[j])

            save_dict(obj, os.path.join(target, f'run_{j + 1}.pkl'))

# Score embeddings, plot embeddings and generate latex tables
score(ID, MODELS, DATASETS)
grid_plot(ID, MODELS, DATASETS, 1)
metrics_train = show_metrics(ID, 'train', MODELS, DATASETS)
metrics_test = show_metrics(ID, 'test', MODELS, DATASETS)
to_latex(ID, 'train', MODELS, DATASETS)
to_latex(ID, 'test', MODELS, DATASETS)
