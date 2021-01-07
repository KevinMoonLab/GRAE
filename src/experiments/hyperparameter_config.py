import scipy.stats as stats

neighbor_param = [5, 10, 20]

PARAM_GRID = {
    'lr': stats.loguniform(a=1e-3, b=1e-2),
    'batch_size': stats.randint(low=32, high=101),
    'weight_decay': [0],
    't': [10, 25, 50, 100, 250],
    'gamma': [0, 1],
    'knn': neighbor_param,
    'n_neighbors': neighbor_param,
    'min_dist': stats.uniform(loc=0, scale=.99),
    'epsilon': stats.loguniform(a=.01, b=250),
    'lam': stats.loguniform(a=1e-1, b=5),
    'margin': stats.loguniform(a=.01, b=100),
}

RANDOM_STATE = 42