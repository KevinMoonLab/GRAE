import scipy.stats as stats

PARAM_GRID = {
    'lr': stats.loguniform(a=.0001, b=.01),
    'batch_size': stats.randint(low=32, high=129),
    'weight_decay': stats.loguniform(a=.01, b=10),
    't': [10, 25, 50, 100, 250],
    'gamma': [0, 1],
    'knn': [5, 10, 20, 100],
    'k': [5, 10, 20, 100],
    'n_neighbors': [5, 10, 20, 100],
    'min_dist': stats.uniform(loc=0, scale=.99),
    'epsilon': stats.loguniform(a=.01, b=250),
    'lam': stats.loguniform(a=.01, b=10),
    'margin': stats.loguniform(a=.01, b=100),
}

RANDOM_STATE = 42