"""Build schedule for test experiments based on hyper parameter search
results. Fetch results directly on Comet servers.
"""
import os
import math
import argparse

import comet_ml
from comet_ml.query import Tag
import pandas as pd
from sklearn.model_selection import ParameterSampler
from grae.experiments.hyperparameter_config import PARAM_GRID, PARAM_GRID_L, RANDOM_STATE

# Parser
parser = argparse.ArgumentParser(
    description='Parse hyper parameter search results and update a target experiment schedule with best '
                'hyper parameters according to validation MSE.')
parser.add_argument('--comet_tag',
                    '-t',
                    help='Comet tag for experiments to consider. Query will look up experiments tagged with '
                         'f\'{-t}_validate\'', type=str)
parser.add_argument('--schedule_path',
                    '-s',
                    help='Schedule path. A file listing all experiments to run with job ids, a copy of which will be'
                         'updated with the best hyperparameters in the search.',
                    type=str,
                    default=None)

parser.add_argument('--results_path',
                    '-r',
                    help='Where to cache results to avoid future Comet queries.',
                    type=str,
                    default=None)
parser.add_argument('--search_archive',
                    '-x',
                    action='store_true',
                    help='Including Comet Archive in search.')
parser.add_argument('--archive',
                    '-a',
                    action='store_true',
                    help='Archive found experiments.')
parser.add_argument('--print',
                    '-p',
                    action='store_true',
                    help='Print resulting DataFrame to console.')

args = parser.parse_args()
if args.comet_tag is None:
    raise Exception('Comet tag is required. Please provide tag to the -t flag.')

if args.schedule_path is None:
    # Used as a reference. A copy will be updated.
    args.schedule_path = os.path.join(os.getcwd(), 'exp_schedule', f'{args.comet_tag}.csv')
if args.results_path is None:
    args.results_path = os.path.join(os.getcwd(), 'results', f'{args.comet_tag}_search.csv')

if not os.path.exists(args.results_path) or args.archive:
    # Fetch results from Comet
    comet_api = comet_ml.api.API()

    # Query all experiments with given tag
    experiments = comet_api.query('chu24', 'grae', Tag(f'{args.comet_tag}_validate'), archived=args.search_archive)

    # Fetch results
    results_dict = dict(
        dataset_name=list(),
        model_name=list(),
        param_no=list(),
        validate_MSE=list(),
        success=list(),
        durationMillis=list(),
    )

    for exp in experiments:
        # Check if experiment was successful. Else continue.
        success_flag = exp.get_others_summary(other='success')

        if len(success_flag) > 0:
            # Parameters
            for n in ('dataset_name', 'model_name'):
                results_dict[n].append(exp.get_parameters_summary(parameter=n)['valueCurrent'])
            # Metrics
            for n in ['validate_MSE']:
                results_dict[n].append(float(exp.get_metrics(metric=n)[0]['metricValue']))
            # Metadata
            for n in ['durationMillis']:
                results_dict[n].append(exp.get_metadata()[n])
            # Others
            for n in ('param_no', 'success'):
                results_dict[n].append(int(exp.get_others_summary(other=n)[0]))

        # Archive if requested
        if args.archive:
            exp.archive()

    # Dataframe
    df = pd.DataFrame.from_dict(results_dict)
    df.to_csv(args.results_path, index=False)

# Load dataframe
df = pd.read_csv(args.results_path, index_col=False)

# Average over folds
df = df.groupby(['dataset_name', 'model_name', 'param_no']).mean()

# Remove hyperaparameters with at least one failed fold
df = df[~df['success'].isnull()].drop(['success'], axis=1)

# Keep hyperparameter number with best validation MSE
idx = df.groupby(['dataset_name', 'model_name'])['validate_MSE'].idxmin()
df = df.loc[idx]

df['durationMin'] = df['durationMillis'] / (1000 * 60)  # Convert to minutes
df['durationMin'] += 1  # Add one minute
df = df.drop(columns='durationMillis')

# Unstack parameter number
df = df.reset_index(level=['param_no'])

# Print dataframe if needed
if args.print:
    print(df.sort_values(['dataset_name', 'validate_MSE']).to_string())

# Convert epochs and duration to ints
df = df.astype(dict(durationMin='int', param_no='int'))

# Load copy of main schedule as reference for new schedule
df_main = pd.read_csv(args.schedule_path, index_col=False)
param_list = list(ParameterSampler(PARAM_GRID,
                                   n_iter=30,
                                   random_state=RANDOM_STATE))
param_list_L = list(ParameterSampler(PARAM_GRID_L, # Parameter grid for iPSC
                                   n_iter=30,
                                   random_state=RANDOM_STATE))


# Helper function to substitute values from reference schedule with the best ones according to the search
def sub_row(x):
    key = x['dataset_name'], x['model_name']
    p_list = param_list_L if x['dataset_name'] == 'IPSC' else param_list

    if key in df.index:
        ref_row = df.loc[key]
        x['estimated_time'] = ref_row['durationMin']
        params = p_list[int(ref_row['param_no'])]

        for key, item in params.items():
            # Only replace hyperparameters if value is already present in schedule
            value = x.get(key)
            if value and not math.isnan(value):
                x[key] = item

    return x


# Substitute hyperparameters from main schedule with best hyper parameters according to validation MSE
df_main = df_main.apply(sub_row, axis=1)
df_main.to_csv(os.path.join(os.path.dirname(args.schedule_path), f'{args.comet_tag}_test.csv'), index=False)
