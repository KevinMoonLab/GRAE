"""Main experiment script to train models and compute metrics. Can also perform a hyper parameter search.

   Experiment schedule should be saved in csv format with lines as experiment and parameters as columns.
   Script runs experiments in schedule marked with -j in the 'job' column.

   All metrics & assets (i.e. embeddings, plot) are saved to Comet.
"""
import os
import argparse

import comet_ml

import pandas as pd

from grae.experiments.experiments import fit_test
from grae.experiments.hyperparameter_config import RANDOM_STATE

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
parser.add_argument('--k_fold',
                    '-k',
                    help='Number of seeds/folds to try. Every run will use a different fold for early stopping and a '
                         'different seed to initialize models. All runs are scored on the same test split.',
                    type=int,
                    default=10)
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

# Get fold and job_id from slurm id
# Cycle through values. args.job can take a value between 0 and (number of jobs * number of folds) - 1.
# Think of it as cycling through the schedule to run all job and fold combinations.
# Validation fold is used mainly for early stopping. Different folds will also use different model seed.
k, job_no = divmod(args.job, n_jobs)
k %= args.k_fold

# Filter schedule
schedule = schedule.loc[schedule['job'] == job_no].drop(['job', 'estimated_time'], 1)
if schedule.shape[0] == 0:
    raise Exception(f'No job marked with the following id : {args.job}.')

# Launch experiments
for _, exp_params in schedule.iterrows():
    params = exp_params.dropna().to_dict()

    # Use same random state for all experiments for reproducibility
    params['random_state'] = RANDOM_STATE

    try:
        fit_test(params,
                 custom_tag=args.comet_tag,
                 data_path=args.data_path,
                 write_path=args.write_path,
                 k=k,
                 others=dict(slurm_id=args.job, k=k, job_no=job_no))
    except Exception as e:
        print(e)
