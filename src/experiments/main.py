"""Main experiment script.

   All results, including embeddings, plots and latex tables will be saved under results/ID in the project root.
   ID is generated based on current time.
"""
import os
import datetime
import time
import copy

import src.data
import src.models
from src.experiments.utils import save_dict
from src.metrics import score
from src.figures import grid_plot, show_metrics
from src.experiments.model_params import DEFAULTS, DATASET_PARAMS

# Fit models
# Models and Datasets for experiment
# Specific model arguments can be changed in the model_params.py module
MODELS = ['GRAE']
DATASETS = ['Faces', 'Teapot']

RUNS = 1  # Number of runs to average
RANDOM_STATES = [36087, 63286, 52270, 10387, 40556, 52487, 26512, 28571, 33380,
                 9369, 28478, 4624, 29114, 41915, 6467, 4216, 16025, 34823,
                 29854, 23853]  # 20 random states. Add more if needed.


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

            # Fetch and split dataset.
            data_train = getattr(src.data, dataset)(split='train', random_state=RANDOM_STATES[j])
            data_test = getattr(src.data, dataset)(split='test', random_state=RANDOM_STATES[j])

            # Get default parameters
            params = copy.deepcopy(DEFAULTS[model])

            if dataset in DATASET_PARAMS[model]:
                params.update(DATASET_PARAMS[model][dataset])

            m = getattr(src.models, model)(random_state=RANDOM_STATES[j], **params)

            # Benchmark fit time
            fit_start = time.time()

            m.fit(data_train)

            fit_stop = time.time()

            fit_time = fit_stop - fit_start

            # Compute reconstruction right away to avoid saving all reconstructed samples
            test_results = m.score(data_test, split_name='test')
            train_results = m.score(data_train, split_name='train')

            # Save embeddings
            obj = dict(fit_time=fit_time,
                       dataset_seed=RANDOM_STATES[j], run_seed=RANDOM_STATES[j],
                       **train_results,
                       **test_results)

            save_dict(obj, os.path.join(target, f'run_{j + 1}.pkl'))

# Score embeddings, plot embeddings and generate latex tables
grid_plot(ID, MODELS, DATASETS, run=1)
score(ID, MODELS, DATASETS)
metrics_train = show_metrics(ID, 'train', MODELS, DATASETS)
metrics_test = show_metrics(ID, 'test', MODELS, DATASETS)

print('Train metrics :')
print(metrics_train)

print('\nTest metrics :')
print(metrics_test)
