"""Utilities."""
from six.moves import cPickle as pickle


def load_dict(filename_):
    """Handy function to load dictionaries.

    Args:
        filename_(str): Name of pickle file.

    Returns:
        dict(dict): Loaded dict.

    """
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def save_dict(di_, filename_):
    """Handy function to save dictionaries.

    Args:
        di_(dict): Dict to save.
        filename_(str): Name of file.
    """
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)
