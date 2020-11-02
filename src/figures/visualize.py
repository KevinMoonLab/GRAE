"""Visualize the embeddings of one run from main.py."""
import os

import matplotlib.pyplot as plt

import src.data
from src.figures.name_maps import get_model_name, get_dataset_name
from src.figures.utils import load_dict

# Plot embeddings
PLOT_RUN = 1


def grid_plot(id, model_list, dataset_list, run, flip=False, full_manifold=False):
    if flip:
        first_dim = model_list
        second_dim = dataset_list
        titles_first_dim = [get_model_name(m) for m in model_list]
        titles_second_dim = [get_dataset_name(m) for m in dataset_list]
    else:
        first_dim = dataset_list
        second_dim = model_list
        titles_first_dim = [get_dataset_name(m) for m in dataset_list]
        titles_second_dim = [get_model_name(m) for m in model_list]

    n_d = len(first_dim)
    n_m = len(second_dim)

    fig, ax = plt.subplots(n_d, n_m, figsize=(n_m * 3.5, n_d * 3.5))
    path = os.path.join(
        os.path.dirname(__file__),
        os.path.join('..', '..', 'results', id)
    )

    for i, first in enumerate(first_dim):
        for j, second in enumerate(second_dim):
            model = first if flip else second
            dataset = second if flip else first
            file_path = os.path.join(path, 'embeddings', model, dataset, f'run_{run}.pkl')

            if os.path.exists(file_path):
                # Retrieve datasets for coloring
                data = load_dict(file_path)
                X_train = getattr(src.data, dataset)(split='train',
                                                     random_state=data['dataset_seed'])
                X_test = getattr(src.data, dataset)(split='test',
                                                    random_state=data['dataset_seed'])
                _, y_train = X_train.numpy()
                _, y_test = X_test.numpy()
                z_train, z_test = data['z_train'], data['z_test']
            else:
                raise Exception('Target embedding does not exist.')

            if n_d == 1 and n_m == 1:
                ax_i = ax
            elif n_d == 1:
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

            if not full_manifold:
                ax_i.scatter(*z_train.T, s=s_train, alpha=.2, color='grey')
                ax_i.scatter(*z_test.T, c=y_test, s=s_test, cmap='jet')
            else:
                ax_i.scatter(*z_train.T, s=s_train, cmap='jet', c=y_train)
                ax_i.scatter(*z_test.T, s=s_train, cmap='jet', c=y_test)

            if i == 0:
                ax_i.set_title(f'{titles_second_dim[j]}', fontsize=25, color='black')

            if j == 0:
                ax_i.set_ylabel(titles_first_dim[i], fontsize=25, color='black')

            ax_i.set_xticks([])
            ax_i.set_yticks([])

    # plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=.0, wspace=.0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(os.path.join(path, f'plot_{run}.png'), bbox_inches='tight', pad_inches=0.035)
    plt.show()
