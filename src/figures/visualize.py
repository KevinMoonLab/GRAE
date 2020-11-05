"""Visualize the embeddings of one run from main.py."""
import os

import matplotlib.pyplot as plt

import src.data
from src.figures.name_getters import get_model_name, get_dataset_name
from src.experiments.utils import load_dict


def grid_plot(id_, model_list, dataset_list, run, flip=False, grayscale_train=True):
    """Plot 2D embeddings of an experiment run.

    Plot both train and test poitns. The layout is a grid with datasets on the vertical axis and models on the
    horizontal one.

    Will display the plots and save the figure as a png under ./results/ID.

    Args:
        id_(str): ID of the desired experiment, as saved under ./results.
        model_list(List[str]): Models to plot.
        dataset_list(List[str]): Datasets to plot.
        run(int): Run to plot
        flip(bool): Switch models to vertical axis.
        grayscale_train(bool): Use grayscale for train points to emphasize test points.

    """
    # Flip dimensions if required
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
        os.path.join('..', '..', 'results', id_)
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

            # Hande cases where there's only one model or one dataset
            if n_d == 1 and n_m == 1:
                ax_i = ax
            elif n_d == 1:
                ax_i = ax[j]
            elif n_m == 1:
                ax_i = ax[i]
            else:
                ax_i = ax[i, j]

            # Make sure test points are larger
            s_train = 1.5
            s_test = 15

            # Smaller points for larger dataset to avoir clutter
            if z_test.shape[0] > 1000:
                s_train /= 10
                s_test /= 10

            if grayscale_train:
                # Grayscale mode
                ax_i.scatter(*z_train.T, s=s_train, alpha=.2, color='grey')
                ax_i.scatter(*z_test.T, c=y_test, s=s_test, cmap='jet')
            else:
                # All points are same
                ax_i.scatter(*z_train.T, s=s_train, cmap='jet', c=y_train)
                ax_i.scatter(*z_test.T, s=s_train, cmap='jet', c=y_test)

            if i == 0:
                # Add column titles
                ax_i.set_title(f'{titles_second_dim[j]}', fontsize=25, color='black')

            if j == 0:
                # Add row titles
                ax_i.set_ylabel(titles_first_dim[i], fontsize=25, color='black')

            ax_i.set_xticks([])
            ax_i.set_yticks([])

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=.0, wspace=.0)

    # Save figure
    plt.savefig(os.path.join(path, f'plot_{run}.png'), bbox_inches='tight', pad_inches=0.035)
    plt.show()
