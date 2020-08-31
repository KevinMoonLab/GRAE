"""Model arguments for the main.py experiments."""
# Some arg dicts that will be reused by various models
# Default PHATE args

DEFAULT_EPOCHS = 800   # First default for most datasets
DEFAULT_EPOCHS_L = 200   # Second default for larger datasets

epoch_dict = dict(  # Dataset specific arguments
    SwissRoll=dict(epochs=DEFAULT_EPOCHS_L),
    Faces=dict(epochs=DEFAULT_EPOCHS),
    RotatedDigits=dict(epochs=DEFAULT_EPOCHS),
    Tracking=dict(epochs=DEFAULT_EPOCHS),
    Teapot=dict(epochs=DEFAULT_EPOCHS),
    Embryoid=dict(epochs=DEFAULT_EPOCHS),
    IPSC=dict(epochs=DEFAULT_EPOCHS_L),
    UMIST=dict(epochs=DEFAULT_EPOCHS),
    COIL100=dict(epochs=DEFAULT_EPOCHS),
)

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
    COIL100=dict(knn=5, t=50),
)


# UMAP neighbors
UMAP_DEFAULTS = dict()
UMAP_dict = dict(  # Dataset specific arguments
    SwissRoll=dict(n_neighbors=20),
    Faces=dict(n_neighbors=15),
    RotatedDigits=dict(n_neighbors=15),
    Tracking=dict(n_neighbors=15),
    Teapot=dict(n_neighbors=15),
    Embryoid=dict(n_neighbors=50),
    IPSC=dict(n_neighbors=50),
    UMIST=dict(n_neighbors=15),
    COIL100=dict(n_neighbors=5),
)

# TSNE perplexity
TSNE_DEFAULTS = dict()
TSNE_dict = dict(  # Dataset specific arguments
    SwissRoll=dict(perplexity=30),
    Faces=dict(perplexity=10),
    RotatedDigits=dict(perplexity=10),
    Tracking=dict(perplexity=10),
    Teapot=dict(perplexity=10),
    Embryoid=dict(perplexity=10),
    IPSC=dict(perplexity=10),
    UMIST=dict(perplexity=10),
    COIL100=dict(perplexity=10),
)

# Add defaults to dataset specific dicts
for key, d in PHATE_dict.items():
    d.update(PHATE_DEFAULTS)
    PHATE_dict[key] = dict(embedder_args=d)  # Wrap under embedder argument
    PHATE_dict[key].update(epoch_dict[key])

# for key, d in UMAP_dict.items():
#     d.update(UMAP_DEFAULTS)
#     UMAP_dict[key] = dict(embedder_args=d)  # Wrap under embedder argument

# for key, d in TSNE_dict.items():
#     d.update(TSNE_DEFAULTS)
#     TSNE_dict[key] = dict(embedder_args=d)  # Wrap under embedder argument

# Model parameters to use for experiments
# Make sure the dict key matches the class name
# Those arguments will be used every time the model is initialized
DEFAULTS = {
    'AE': dict(),
    'GRAE': dict(),
    'SGRAE': dict(),
    'UMAP': dict(),
    'EAERMargin': dict(),
    'TopoAE': dict(),
}

DATASET_PARAMS = {
    'AE': epoch_dict,
    'GRAE': PHATE_dict,
    'SGRAE': PHATE_dict,
    'UMAP': UMAP_dict,
    'EAERMargin': epoch_dict,
    'TopoAE': epoch_dict,
}

