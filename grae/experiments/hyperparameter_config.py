import scipy.stats as stats

neighbor_param = [5, 10, 20]

PARAM_GRID = {
    'lr': stats.loguniform(a=2e-4, b=2e-3),
    'batch_size': stats.randint(low=32, high=101),
    'weight_decay': [0],
    't': [10, 25, 50, 100, 250],
    'gamma': [0, 1],
    'knn': neighbor_param,
    'n_neighbors': neighbor_param,
    'min_dist': stats.uniform(loc=0, scale=.99),
    'epsilon': stats.loguniform(a=.01, b=250),
    'lam': stats.loguniform(a=1e-2, b=1e1),
    'margin': stats.loguniform(a=.01, b=10),
}

# Main seed. Used for train/test splitting and generating hyperparameter combinations
RANDOM_STATE = 42

# Ancillary seed for sampling validation splits and seeding models
FOLD_SEEDS = [4837, 2963, 1504, 6387, 5865, 9969, 5313, 2421, 2524, 5511, 4370, 2677, 5785, 7720, 2108, 1055, 6533,
              9591, 3344, 6267, 1650, 7304, 7168, 3115, 1215, 9499, 7472, 7297,  771,  842, 2652,  148, 2987, 5441,
              3643, 7822,  396, 7544, 9707, 8279, 4594, 5374, 5028, 1422, 7866, 6271, 4957, 7053, 9060, 1268]
