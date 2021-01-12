"""Build schedule for complete experiment based on hyper parameter search
results.

"""
import os
import math

import pandas as pd
from sklearn.model_selection import ParameterSampler
from src.experiments.hyperparameter_config import PARAM_GRID, RANDOM_STATE

SEARCH_RESULTS = os.path.join('..', 'results', 'search_results_jan_9.csv')
MAIN_SCHEDULE = os.path.join('..', 'exp_schedule', 'main.csv')  # Use as reference to generate new schedule

df = pd.read_csv(SEARCH_RESULTS, index_col=False)

# Keep only required columns
df = df[['dataset_name', 'model_name', 'param_no', 'validate_MSE', 'success', 'Duration', 'Epochs']]

# Average over folds
df = df.groupby(['dataset_name', 'model_name', 'param_no']).mean()

# Remove hyperaparameters with at least one failed fold
df = df[~df['success'].isnull()].drop(['success'], axis=1)

# Keep hyperparameter number with best validation MSE
idx = df.groupby(['dataset_name', 'model_name'])['validate_MSE'].idxmin()
df = df.loc[idx]

df['Duration'] /= 60  # Convert to minutes
df['Duration'] += 1  # Add one minute

# Unstack parameter number
df = df.reset_index(level=['param_no'])

# Convert epochs and duration to ints
df = df.astype(dict(Epochs='int', Duration='int', param_no='int'))

# Load copy of main schedule as reference for new schedule
df_main = pd.read_csv(MAIN_SCHEDULE, index_col=False)
param_list = list(ParameterSampler(PARAM_GRID,
                                   n_iter=30,
                                   random_state=RANDOM_STATE))


def sub_row(x):
    key = x['dataset_name'], x['model_name']
    if key in df.index:
        ref_row = df.loc[key]
        x['estimated_time'] = ref_row['Duration']
        x['epochs'] = ref_row['Epochs']
        params = param_list[int(ref_row['param_no'])]

        for key, item in params.items():
            # Only replace hyperparameters if value is already present in schedule
            if not math.isnan(x[key]):
                x[key] = item

    return x


# Substitute hyperparameters from main schedule with best hyper parameters according to validation MSE
df_main = df_main.apply(sub_row, axis=1)
df_main.to_csv(os.path.join('..', 'exp_schedule', 'main_jan_9.csv'), index=False)
