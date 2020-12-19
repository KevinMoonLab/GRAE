"""Script to benchmark fit and transform time performance on dataset of increasing size.

   Results are saved under the results folder time_experiment/{generated-ID}.csv
   TODO: Update for new Comet integration.
"""
import os
import datetime
import time
import copy

import numpy as np

import src.data
import src.models
from src.experiments.model_params import DEFAULTS, DATASET_PARAMS, N_COMPONENTS, RANDOM_STATES
from src.metrics import Book

# Experiment parameters
RUNS = 3
MODELS = ['GRAE', 'AE', 'UMAP']
DATASETS = ['Tracking']
N_SPACE = np.arange(800, 2000 + 1, step=200)

# Tracked variable (in addition to model, dataset, run no & split, which are tracked by default)
METRICS = ['n_sample', 'fit_time', 'transform_time', 'rec_time', 'MSE_train', 'MSE_test']

# Create PATH variable and directory to save embeddings
# Generate ID and save results in the results folder
ID = str(int((datetime.datetime.now() - datetime.datetime(2020, 8, 1)).total_seconds() * 1e6))  # Create ID with time
PATH = os.path.join(
    os.path.dirname(__file__),
    os.path.join('..', '..', 'results', 'time_experiment')
)
if not os.path.exists(PATH):
    os.makedirs(PATH)

FILE_PATH = os.path.join(PATH, ID + '.csv')


# Insert warm up run, otherwise first run always seem to take more time
N_SPACE = np.insert(N_SPACE, 0, 50)

# Object to save results
book = Book(csv_file_path=FILE_PATH, datasets=DATASETS, models=MODELS, metrics=METRICS)

# Main experiment loop
for k, n_sample in enumerate(N_SPACE):
    print(f'Training on {n_sample} samples...')
    for model in MODELS:
        print(f'    Training {model}...')

        for i, dataset in enumerate(DATASETS):

            print(f'       On {dataset}...')

            # Training loop
            for j in range(RUNS):
                print(f'       Run {j + 1}...')

                # Fetch and split dataset.
                # Data used for training
                data_train = getattr(src.data, dataset)(split='train', random_state=RANDOM_STATES[j])

                # Data used for testing transform & reconstruction times
                # Benchmark transform and reconstruction time on new data to make sure models don't simply return an
                # embedding precomputed during training (some manifold learning implementations may use that trick)
                data_test_time = getattr(src.data, dataset)(split='none', random_state=RANDOM_STATES[j])

                # Use fixed size test split to benchmark MSE
                data_test_MSE = getattr(src.data, dataset)(split='test', random_state=RANDOM_STATES[j])

                # Subsample datasets
                data_train = data_train.random_subset(n=n_sample, random_state=RANDOM_STATES[j])
                data_test_time = data_test_time.random_subset(n=n_sample, random_state=RANDOM_STATES[j])

                if len(data_test_MSE) > 5000:
                    data_test_MSE = data_test_MSE.random_subset(n=5000, random_state=RANDOM_STATES[j])

                # Get default parameters
                params = copy.deepcopy(DEFAULTS[model])

                if dataset in DATASET_PARAMS[model]:
                    params.update(DATASET_PARAMS[model][dataset])

                m = getattr(src.models, model)(n_components=N_COMPONENTS, random_state=RANDOM_STATES[j], **params)

                # Benchmark fit time
                fit_start = time.time()

                m.fit(data_train)

                fit_stop = time.time()

                fit_time = fit_stop - fit_start

                test_results = m.score(data_test_time, split_name='test')
                train_results = m.score(data_train, split_name='train')
                test_results_MSE = m.score(data_test_MSE, split_name='test')

                # Save results (ignore warm up run)
                if k > 0:
                    book.add_entry(model=model,
                                   dataset=dataset,
                                   run=j,
                                   split='none',
                                   n_sample=n_sample,
                                   fit_time=fit_time,
                                   transform_time=test_results['transform_time_test'],
                                   rec_time=test_results['rec_time_test'],
                                   MSE_train=train_results['MSE_train'],
                                   MSE_test=test_results_MSE['MSE_test'])

    # Save all results to csv for a given n_sample
    book.save_to_csv()
