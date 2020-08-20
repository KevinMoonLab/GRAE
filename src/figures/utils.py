"""Utils for experiments."""
from six.moves import cPickle as pickle  # for performance

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

