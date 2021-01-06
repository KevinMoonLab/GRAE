"""Build schedule for complete experiment based on hyper parameter search
results.

"""
import os
import math

import pandas as pd
from sklearn.model_selection import ParameterSampler
from src.experiments.hyperparameter_config import PARAM_GRID, RANDOM_STATE

df = pd.read_csv(os.path.join('..', 'results', 'search_results_jan_4.csv'), index_col=False)

# If early stopped, use that epoch count instead of default epoch argument
es = ~df['train_early_stopped'].isnull()
df.loc[es, 'Epochs'] = df['train_early_stopped'][es]

# Drop some colums
df = df.drop(['Name', 'job', 'k_fold', 'train_early_stopped'], axis=1)

# Average over folds
df = df.groupby(['model_name', 'dataset_name', 'param_no']).mean()

# Remove hyperaparameters with at least one failed fold
df = df[~df['success'].isnull()].drop(['success'], axis= 1)

# Keep hyperparameter number with best validation MSE
idx = df.groupby(['model_name', 'dataset_name'])['validate_MSE'].idxmin()
df = df.loc[idx]

df['Duration'] /= 60  # Convert to minutes
df['Duration'] *= 1.18  # Inflate to account for larger train set
df['Duration'] += 1  # Add one minute


# Unstack parameter number
df = df.reset_index(level=['param_no'])

# Convert epochs and duration to ints
df = df.astype(dict(Epochs='int', Duration='int', param_no='int'))

print(df)

# Load copy of main schedule
df_main = pd.read_csv(os.path.join('..', 'exp_schedule', 'main.csv'), index_col=False)
param_list = list(ParameterSampler(PARAM_GRID,
                                   n_iter=30,
                                   random_state=RANDOM_STATE))

def sub_row(x):
    key = x['model_name'], x['dataset_name']
    if key in df.index:
        ref_row = df.loc[key]
        x['estimated_time'] = ref_row['Duration']
        x['epochs'] = ref_row['Epochs']
        params = param_list[int(ref_row['param_no'])]

        for key, item in params.items():
            if not math.isnan(x[key]):
                x[key] = item

    return x

df_main = df_main.apply(sub_row, axis=1)
df_main.to_csv(os.path.join('..', 'exp_schedule', 'main_search.csv'), index=False)
