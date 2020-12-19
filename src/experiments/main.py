"""Main experiment script.

   Run full experiments in SCHEDULE_NAME marked with JOB_ID.
   Experiment schedule should be saved in csv format under exp_schedule in the root folder.
   All metrics & assets (i.e. embeddings, plot) are saved to Comet.
"""
import sys
import os

import pandas as pd

from src.experiments.experiments import fit_test

# Fit models
# Models and Datasets for experiment

SCHEDULE_NAME = 'main'
JOB_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CUSTOM_TAG = sys.argv[1] if len(sys.argv) > 1 else 'no_tag'

# Get Schedule
PATH = os.path.join(
    os.path.dirname(__file__),
    os.path.join('..', '..', 'exp_schedule')
)

# Read schedule and only keep experiment tagged with current job
schedule = pd.read_csv(os.path.join(PATH, SCHEDULE_NAME + '.csv'))
schedule = schedule.loc[schedule['job'] == JOB_ID].drop('job', 1)

for _, exp_params in schedule.iterrows():
    params = exp_params.dropna().to_dict()
    fit_test(params, custom_tag=CUSTOM_TAG)

