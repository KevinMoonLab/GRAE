"""Script to test the impact of latent space dimension on reconstruction quality (MSE).

   Results are saved under the results folder dimension_experiment/{generated-ID}.csv

   TODO: Update for new Comet integration.
"""
import os
import datetime
import copy

import numpy as np

import src.data
import src.models
from src.experiments.model_params import DEFAULTS, DATASET_PARAMS, N_COMPONENTS, RANDOM_STATES
from src.metrics import Book

# Experiment parameters
RUNS = 1
MODELS = ['AE', 'GRAE']
DATASETS = ['Tracking', 'Teapot', 'Faces']
DIM_SPACE = np.arange(1, 11)

# Tracked variable (in addition to model, dataset, run no & split, which are tracked by default)
METRICS = ['dim', 'MSE_train', 'MSE_test']

# Create PATH variable and directory to save embeddings
# Generate ID and save results in the results folder
ID = str(int((datetime.datetime.now() - datetime.datetime(2020, 8, 1)).total_seconds() * 1e6))  # Create ID with time
PATH = os.path.join(
    os.path.dirname(__file__),
    os.path.join('..', '..', 'results', 'dimension_experiment')
)
if not os.path.exists(PATH):
    os.makedirs(PATH)

FILE_PATH = os.path.join(PATH, ID + '.csv')


# Object to save results
book = Book(csv_file_path=FILE_PATH, datasets=DATASETS, models=MODELS, metrics=METRICS)

# Main experiment loop
for i, dataset in enumerate(DATASETS):
    print(f'Training on {dataset}...')
    for k, dim in enumerate(DIM_SPACE):
        print(f'    Training on {dim} dimensions...')
        for model in MODELS:
            print(f'        Training {model}...')

            # Training loop
            for j in range(RUNS):
                print(f'         Run {j + 1}...')

                # Fetch and split datasets
                # Data used for training
                data_train = getattr(src.data, dataset)(split='train', random_state=RANDOM_STATES[j])
                data_test = getattr(src.data, dataset)(split='test', random_state=RANDOM_STATES[j])

                # Get default parameters
                params = copy.deepcopy(DEFAULTS[model])

                if dataset in DATASET_PARAMS[model]:
                    params.update(DATASET_PARAMS[model][dataset])

                m = getattr(src.models, model)(n_components=dim, random_state=RANDOM_STATES[j], **params)

                m.fit(data_train)

                # Score models
                test_results = m.score(data_test, split_name='test')
                train_results = m.score(data_train, split_name='train')

                # Save results
                book.add_entry(model=model,
                               dataset=dataset,
                               run=j,
                               split='none',
                               dim=dim,
                               MSE_train=train_results['MSE_train'],
                               MSE_test=test_results['MSE_test'])

        # Save all results to csv for when all models were trained for a given dimension
        book.save_to_csv()
