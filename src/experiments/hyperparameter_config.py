import scipy.stats as stats

PARAM_GRID = {
    'lr': stats.loguniform(a=1e-3, b=1e-2),
    'batch_size': stats.randint(low=32, high=129),
    'weight_decay': stats.loguniform(a=1e-7, b=1e-3),
    't': [10, 25, 50, 100, 250],
    'gamma': [0, 1],
    'knn': [5, 10, 20, 100],
    'k': [5, 10, 20, 100],
    'n_neighbors': [5, 10, 20, 100],
    'min_dist': stats.uniform(loc=0, scale=.99),
    'epsilon': stats.loguniform(a=.01, b=250),
    'lam': stats.loguniform(a=1e-1, b=1e2),
    'margin': stats.loguniform(a=.01, b=100),
}

RANDOM_STATE = 42