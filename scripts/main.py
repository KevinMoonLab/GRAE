"""Main experiment script to train models and compute metrics. Can also perform a hyper parameter search.

   Experiment schedule should be saved in csv format with lines as experiment and parameters as columns.
   Script runs experiments in schedule marked with -j in the 'job' column.

   If the --hyper flag is set, parameters are resampled to perform a search. -j can then range from
   0 to (number of experiments * number of cross validation folds * n_iter) - 1 to iterate through every possible
   combinations of jobs, folds and hyper parameters. Useful to launch jobs with Slurm Arrays.

   All metrics & assets (i.e. embeddings, plot) are saved to Comet.
"""
import os
import argparse

import comet_ml

import pandas as pd
from sklearn.model_selection import ParameterSampler

from src.experiments.experiments import fit_test, fit_validate

# Parser
parser = argparse.ArgumentParser(description='Run full experiments with specific hyper parameters '
                                             'as scheduled in a .csv file...')
parser.add_argument('--comet_tag', '-t',
                    help='Comet tag for experiment', type=str, default='no_tag')
parser.add_argument('--job', '-j',
                    help='Job id. Run all experiment in schedule with this number.', type=int, default=0)
parser.add_argument('--schedule_path',
                    '-s',
                    help='Schedule path. A file listing all experiments to run with associated ids.',
                    type=str,
                    default=os.path.join(os.getcwd(), 'exp_schedule', 'main.csv'))
parser.add_argument('--data_path',
                    '-d',
                    help='Data path. Otherwise assumes a \'data\' folder in current working directory.',
                    type=str,
                    default=os.path.join(os.getcwd(), 'data'))
parser.add_argument('--write_path',
                    '-w',
                    help='Where to write temp files. Otherwise assumes current working directory.',
                    type=str,
                    default=os.getcwd())
parser.add_argument('--k_fold',
                    '-k',
                    help='Number of folds for cross validation.',
                    type=int,
                    default=3)
parser.add_argument('--n_iter',
                    '-n',
                    help='Number of combinations to try.',
                    type=int,
                    default=20)
parser.add_argument('--hyper',
                    help='Replace metrics from schedule with sampled parameters to perform hyper parameter search',
                    action="store_true")

args = parser.parse_args()

# Hyperparameter grid
PARAM_GRID = {
    'lr': [.001, .0001, .00001],
    'batch_size': [32, 64, 128],
    'weight_decay': [.01, .1, 1, 10],
    't': [10, 50, 100, 250],
    'knn': [10, 15, 20, 100],
    'n_neighbors': [10, 15, 20, 100],
    'min_dist': [.1, .3, .5],
    'epsilon': [10, 50, 100, 250],
    'lam': [.1, 1, 10, 100],
    'margin': [.1, 1, 10],
}

# Get Schedule
# Read schedule and only keep experiment tagged with current job
schedule = pd.read_csv(args.schedule_path)
n_jobs = schedule['job'].max() + 1  # Number of different jobs

# Get fold, absolute job_id and parameter combination number
# Cycle through values so calling args.job between 0 and (number of jobs * number of folds * number of hyper
# parameter combinations) - 1 will cover all possibilities. Think of it as duplicating the schedule for all
# fold and parameter combination possibilities
k, job_no = divmod(args.job, n_jobs)
k %= args.k_fold
param_no = int(args.job / (n_jobs * args.k_fold))

# Filter schedule
schedule = schedule.loc[schedule['job'] == job_no].drop(['job', 'estimated_time'], 1)

# Launch experiments
for _, exp_params in schedule.iterrows():
    params = exp_params.dropna().to_dict()

    # Use same random state for all experiments for reproducibility
    params['random_state'] = 42

    if args.hyper:
        # Replace epochs by max_epochs when doing validation. Early stopping will be used.
        # Useful to get an idea of the number of epochs to use when training on the full training set
        if 'epochs' in params.keys():
            params['epochs'] = params.pop('max_epochs')

        # Fetch parameter combination
        sampled_params = list(ParameterSampler(PARAM_GRID, n_iter=args.n_iter, random_state=42))[param_no]

        # Only keep parameters relevant to given model
        params.update((key, sampled_params[key]) for key in params.keys() & sampled_params.keys())

        try:
            fit_validate(params, custom_tag=args.comet_tag, data_path=args.data_path, k=k, write_path=args.write_path)
        except Exception as e:
            print(e)
    else:
        # Training on full dataset and compute metrics on test split
        if 'epochs' in params.keys():
            params.pop('max_epochs')  # Not needed when training on full train split. Use directly epochs instead

        try:
            fit_test(params, custom_tag=args.comet_tag, data_path=args.data_path, write_path=args.write_path)
        except Exception as e:
            print(e)
