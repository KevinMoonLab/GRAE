"""Main experiment script.

   Run full experiments in SCHEDULE_NAME marked with JOB_ID.
   Experiment schedule should be saved in csv format under exp_schedule in the root folder.
   All metrics & assets (i.e. embeddings, plot) are saved to Comet.
"""
import os
import argparse

import pandas as pd

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
parser.add_argument('--validation',
                    help='Compute metrics on validation split. Turn off logging of assets and metrics after every epoch '
                         'for faster training.',
                    action="store_true")

args = parser.parse_args()


# Get Schedule
# Read schedule and only keep experiment tagged with current job
schedule = pd.read_csv(args.schedule_path)
schedule = schedule.loc[schedule['job'] == args.job].drop(['job', 'estimated_time'], 1)

# Launch experiments
for _, exp_params in schedule.iterrows():
    params = exp_params.dropna().to_dict()

    # Use same random state for all experiments for reproducibility
    params['random_state'] = 42

    if args.validation:
        # Replace epochs by max_epochs when doing validation. Early stopping will be used.
        # Useful to get an idea of the number of epochs to use when training on the full training set
        if 'epochs' in params.keys():
            params['epochs'] = params.pop('max_epochs')

        # Get fold
        k = params.pop('k')

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

