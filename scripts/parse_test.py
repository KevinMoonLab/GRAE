"""Aggregate test results and plot box plots of all runs. Fetch results directly on comet servers."""
import os
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import comet_ml
from comet_ml.query import Tag
import pandas as pd

# Parser
parser = argparse.ArgumentParser(
    description='Aggregate test results and plot box plots of all runs.')
parser.add_argument('--comet_tag',
                    '-t',
                    help='Comet tag for experiments to consider. Query will look up experiments tagged with '
                         'f\'{-t}_validate\'', type=str)
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

if args.results_path is None:
    args.results_path = os.path.join(os.getcwd(), 'results', f'{args.comet_tag}_test.csv')

if not os.path.exists(args.results_path) or args.archive:
    # Fetch results from Comet
    comet_api = comet_ml.api.API()

    # Query all experiments with given tag
    experiments = comet_api.query('chu24', 'grae', Tag(f'{args.comet_tag}_test'), archived=args.search_archive)

    # Fetch results
    results_dict = dict(
        dataset_name=list(),
        model_name=list(),
        test_MSE=list(),
        test_R2=list(),
        test_Acc=list(),
        success=list(),
    )

    for exp in experiments:
        # Check if experiment was successful. Else continue.
        success_flag = exp.get_others_summary(other='success')

        if len(success_flag) > 0:
            # Parameters
            for n in ('dataset_name', 'model_name'):
                results_dict[n].append(exp.get_parameters_summary(parameter=n)['valueCurrent'])
            # Metrics
            for n in ['test_MSE', 'test_R2', 'test_Acc']:
                results_dict[n].append(float(exp.get_metrics(metric=n)[0]['metricValue']))
            # Others
            for n in ['success']:
                results_dict[n].append(int(exp.get_others_summary(other=n)[0]))

        # Archive if requested
        if args.archive:
            exp.archive()

    # Dataframe
    df = pd.DataFrame.from_dict(results_dict)
    df.to_csv(args.results_path, index=False)

# Load dataframe
df = pd.read_csv(args.results_path, index_col=False)

df = df.loc[df['success'] == 1].drop(columns='success', axis=1)  # Filter out unsuccessful experiments
# df = df.loc[df['model_name'] != 'DiffusionNet']

df_agg = df.groupby(['dataset_name', 'model_name']).mean().sort_values(['dataset_name', 'test_MSE'])

if args.print:

    for metric in ['test_R2', 'test_MSE', 'test_Acc']:
        print(f'Results ordered by {metric}:\n')
        print(df_agg.sort_values(['dataset_name', metric]).to_string())
        print('\n\n')
        g = sns.catplot(x=metric, y="model_name", col="dataset_name",
                        data=df, kind="violin", split=False, scale='width', sharex=False, col_wrap=4,
                        order=['GRAE', 'GRAEUMAP', 'AE', 'EAERMargin', 'TopoAE', 'DiffusionNet'],
                        cut=0,
                        size=2,
                        height=4, aspect=1.2)

        g.set_titles(col_template="{col_name}", fontweight='bold', size=15)

        plt.show()
