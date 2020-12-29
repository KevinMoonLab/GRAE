"""Main experiment script to train models and compute metrics. Can also perform a hyper parameter search.

   Experiment schedule should be saved in csv format with lines as experiment and parameters as columns.
   Script runs experiments in schedule marked with -j in the 'job' column.

   All metrics & assets (i.e. embeddings, plot) are saved to Comet.
"""
import os
import argparse

import comet_ml

import pandas as pd

from src.experiments.experiments import fit_test

RANDOM_STATE = 42  # Main seed for models, dataset splits and hyper parameter search

# Parser
parser = argparse.ArgumentParser(description='Run full experiments with specific hyper parameters '
                                             'as scheduled in a .csv file...')
parser.add_argument('--comet_tag', '-t',
                    help='Comet tag for experiment', type=str, default='no_tag')
parser.add_argument('--job', '-j',
                    help='Slurm id. Run all experiment in schedule marked with this number.',
                    type=int,
                    default=0)
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

args = parser.parse_args()

# Get Schedule
# Read schedule and only keep experiment tagged with current job
schedule = pd.read_csv(args.schedule_path)
n_jobs = schedule['job'].max() + 1  # Number of different jobs

# Filter schedule
schedule = schedule.loc[schedule['job'] == args.job].drop(['job', 'estimated_time'], 1)
if schedule.shape[0] == 0:
    raise Exception(f'No job marked with the following id : {args.job}.')

# Launch experiments
for _, exp_params in schedule.iterrows():
    params = exp_params.dropna().to_dict()

    # Use same random state for all experiments for reproducibility
    params['random_state'] = RANDOM_STATE

    # Training on full dataset and compute metrics on test split
    if 'epochs' in params.keys():
        params.pop('max_epochs')  # Not needed when training on full train split. Use directly epochs instead

    try:
        fit_test(params,
                 custom_tag=args.comet_tag,
                 data_path=args.data_path,
                 write_path=args.write_path,
                 others=dict(slurm_id=args.job))
    except Exception as e:
        print(e)
