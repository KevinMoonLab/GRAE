"""Visualize the embeddings of one run from main.py."""
import os

import matplotlib.pyplot as plt

import src.data
from src.figures.name_maps import get_model_name
from src.figures.utils import load_dict

# Plot embeddings
PLOT_RUN = 1


def grid_plot(id, model_list, dataset_list, run):
    titles = [get_model_name(m) for m in model_list]
    n_d = len(dataset_list)
    n_m = len(model_list)
    fig, ax = plt.subplots(n_d, n_m, figsize=(n_m * 3.5, n_d * 3.5))
    path = os.path.join(
        os.path.dirname(__file__),
        os.path.join('..', '..', 'results', id)
    )

    for j, model in enumerate(model_list):
        for i, dataset in enumerate(dataset_list):
            file_path = os.path.join(path, 'embeddings', model, dataset, f'run_{run}.pkl')

            if os.path.exists(file_path):
                # Retrieve datasets for coloring
                data = load_dict(file_path)
                X_train = getattr(src.data, dataset)(split='train',
                                                     seed=data['dataset_seed'])
                X_test = getattr(src.data, dataset)(split='test',
                                                    seed=data['dataset_seed'])
                _, y_train = X_train.numpy()
                _, y_test = X_test.numpy()
                z_train, z_test = data['z_train'], data['z_test']
            else:
                raise Exception('Target embedding does not exist.')

            if n_d == 1:
                ax_i = ax[j]
            elif n_m == 1:
                ax_i = ax[i]
            else:
                ax_i = ax[i, j]

            s_train = 1.5

            s_test = 15

            if z_test.shape[0] > 1000:
                s_train /= 10
                s_test /= 10

            ax_i.scatter(*z_train.T, s=s_train, alpha=.2, color='grey')

            ax_i.scatter(*z_test.T, c=y_test, s=s_test, cmap='jet')

            if i == 0:
                ax_i.set_title(f'{titles[j]}', fontsize=20, color='black')
            ax_i.set_xticks([])
            ax_i.set_yticks([])

    plt.savefig(os.path.join(path, f'plot_{run}.png'))
    plt.show()
