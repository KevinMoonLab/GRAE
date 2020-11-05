"""Model arguments for the main.py experiments."""
# Some arg dicts that will be reused by various models
DEFAULT_EPOCHS = 4  # First default for most datasets
DEFAULT_EPOCHS_L = DEFAULT_EPOCHS // 4   # Second default for larger datasets

# Epoch dict used by all AE-based models
epoch_dict = dict(  # Dataset specific arguments
    SwissRoll=dict(epochs=DEFAULT_EPOCHS_L),
    Faces=dict(epochs=DEFAULT_EPOCHS),
    RotatedDigits=dict(epochs=DEFAULT_EPOCHS),
    Tracking=dict(epochs=DEFAULT_EPOCHS),
    Teapot=dict(epochs=DEFAULT_EPOCHS),
    Embryoid=dict(epochs=DEFAULT_EPOCHS),
    IPSC=dict(epochs=DEFAULT_EPOCHS_L),
    UMIST=dict(epochs=DEFAULT_EPOCHS),
)

# PHATE parameters for GRAE
PHATE_DEFAULTS = dict(verbose=0, n_jobs=-1)

PHATE_dict = dict(  # Dataset specific arguments
    SwissRoll=dict(knn=20, t=50),
    Faces=dict(knn=5, t=50),
    RotatedDigits=dict(knn=5, t=50),
    Tracking=dict(knn=15, t=50),
    Teapot=dict(knn=5, t=50),
    Embryoid=dict(knn=15, t=50),
    IPSC=dict(knn=15, t=250),
    UMIST=dict(knn=5, t=50),
)

for key, d in PHATE_dict.items():
    # Merge dicts and update with epochs
    d.update(PHATE_DEFAULTS)
    PHATE_dict[key] = dict(embedder_args=d)  # Wrap under embedder argument
    PHATE_dict[key].update(epoch_dict[key])

# UMAP parameters
UMAP_DEFAULTS = dict(min_dist=.1)

UMAP_dict = dict(  # Dataset specific arguments
    SwissRoll=dict(n_neighbors=20),
    Faces=dict(n_neighbors=15),
    RotatedDigits=dict(n_neighbors=15),
    Tracking=dict(n_neighbors=15),
    Teapot=dict(n_neighbors=15),
    Embryoid=dict(n_neighbors=50),
    IPSC=dict(n_neighbors=50),
    UMIST=dict(n_neighbors=15),
)

GRAEUMAP_dict = dict()

for key, d in UMAP_dict.items():
    # Merge defaults and specific and update with epochs
    d.update(UMAP_DEFAULTS)
    GRAEUMAP_dict[key] = dict(embedder_args=d)  # Wrap under embedder argument
    GRAEUMAP_dict[key].update(epoch_dict[key])


# Dict for Diffusion net
subsample_dict = dict(
    SwissRoll=dict(subsample=None),
    Faces=dict(subsample=None),
    RotatedDigits=dict(subsample=None),
    Tracking=dict(subsample=725),  # 725
    Teapot=dict(subsample=280),
    Embryoid=dict(subsample=None),
    IPSC=dict(subsample=35000),
    UMIST=dict(subsample=250),  # 250
)

# Diffusion Nets
DN_dict = dict(  # Dataset specific arguments
    SwissRoll=dict(n_neighbors=100),
    Faces=dict(n_neighbors=15),
    RotatedDigits=dict(n_neighbors=200, epsilon=50),
    Tracking=dict(n_neighbors=10),
    Teapot=dict(n_neighbors=15),
    Embryoid=dict(n_neighbors=100),
    IPSC=dict(n_neighbors=200, epsilon=50),
    UMIST=dict(n_neighbors=100, epsilon=100),
)

for key, d in DN_dict.items():
    DN_dict[key].update(subsample_dict[key])
    DN_dict[key].update(epoch_dict[key])


# Model parameters to use for experiments
# Make sure the dict key matches the class name
# Those arguments will be used every time the model is initialized
DEFAULTS = {
    'AE': dict(),
    'GRAE': dict(),
    'SmallGRAE': dict(),
    'LargeGRAE': dict(),
    'UMAP': dict(),
    'GRAEUMAP' : dict(),
    'DiffusionNet': dict(),
    'EAERMargin': dict(),
    'TopoAE': dict(),
}

# These arguments are dataset specific
DATASET_PARAMS = {
    'AE': epoch_dict,
    'GRAE': PHATE_dict,
    'SmallGRAE': PHATE_dict,
    'LargeGRAE': PHATE_dict,
    'UMAP': UMAP_dict,
    'GRAEUMAP' : GRAEUMAP_dict,
    'DiffusionNet': DN_dict,
    'EAERMargin': epoch_dict,
    'TopoAE': epoch_dict,
}

