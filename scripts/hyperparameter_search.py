"""Hyperparameter search script.

   Experiment schedule should be saved in csv format with lines as experiment and parameters as columns.

   Slurm id (option -j) can range from 0 to (number of jobs in schedule *
   k_fold * n_iter) - 1 to iterate through every possible combinations of jobs,
   folds and hyper parameters. Useful to launch jobs with Slurm Arrays.

   All metrics are saved to Comet.
"""
import os
import argparse

import comet_ml
import pandas as pd
from sklearn.model_selection import ParameterSampler

from grae.experiments.hyperparameter_config import PARAM_GRID, PARAM_GRID_L, RANDOM_STATE
from grae.experiments.experiments import fit_validate

# Parser
parser = argparse.ArgumentParser(
    description='Run full experiments with resampled hyper parameters '
                'as scheduled in a .csv file...')
parser.add_argument('--comet_tag', '-t',
                    help='Comet tag for experiment', type=str,
                    default='no_tag')
parser.add_argument('--job', '-j',
                    help='Slurm id. Run combination of experiments, fold and parameters corresponding'
                         'to this number.', type=int, default=0)
parser.add_argument('--schedule_path',
                    '-s',
                    help='Schedule path. A file listing all experiments to run with associated ids.',
                    type=str,
                    default=os.path.join(os.getcwd(), 'exp_schedule',
                                         'main.csv'))
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
                    help='Number of hyperparameter combinations to try.',
                    type=int,
                    default=20)

args = parser.parse_args()

# Hyperparameter grid

# Get Schedule
# Read schedule and only keep experiment tagged with current job
schedule = pd.read_csv(args.schedule_path)
n_jobs = schedule['job'].max() + 1  # Number of different jobs

# Get fold, job_id and parameter combination from slurm id
# Cycle through values. args.job can take a value between 0 and (number of jobs * number of folds * number of hyper
# parameter combinations) - 1. Think of it as cycling through the schedule to run all
# job, fold and parameter combinations.
k, job_no = divmod(args.job, n_jobs)
k %= args.k_fold
param_no = int(args.job / (n_jobs * args.k_fold))

# Filter schedule
schedule = schedule.loc[schedule['job'] == job_no].drop(
    ['job', 'estimated_time'], 1)
if schedule.shape[0] == 0:
    raise Exception(f'No job marked with the following id : {job_no}.')

# Launch experiments
for _, exp_params in schedule.iterrows():
    params = exp_params.dropna().to_dict()

    # Use same random state for all experiments for reproducibility
    # Note : this seed is used for splitting the data. Ancillary seeds are used to initialize models
    params['random_state'] = RANDOM_STATE

    # Fetch parameter combination
    # Use larger neighborhood parameters for iPSC
    grid = PARAM_GRID_L if params['dataset_name'] == 'IPSC' else PARAM_GRID
    param_list = list(ParameterSampler(grid,
                                       n_iter=args.n_iter,
                                       random_state=RANDOM_STATE))
    sampled_params = param_list[param_no]

    # Only keep parameters relevant to given model
    params.update((key, sampled_params[key]) for key in
                  params.keys() & sampled_params.keys())

    try:
        fit_validate(params,
                     custom_tag=args.comet_tag,
                     data_path=args.data_path,
                     k=k,
                     others=dict(slurm_id=args.job,  # Actual job called
                                 job=job_no,  # Inferred job id in schedule
                                 k_fold=k,
                                 param_no=param_no),
                     write_path=args.write_path)
    except Exception as e:
        print(e)
