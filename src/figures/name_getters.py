"""Prettier names for models, datasets and metrics."""
model_name = dict(UMAP='UMAP',
                  diffusion_net='Diffusion Nets',
                  PHATE='PHATE',
                  TSNE='t-SNE',
                  TopoAE='TAE',
                  EAERMargin='EAER-Margin',
                  SmallGRAE='λ = 0.1',
                  LargeGRAE='λ = 100',
                  DiffusionNet='Diffusion Nets',
                  GRAE='GRAE',
                  GRAE_100='GRAE',
                  GRAEUMAP='GRAE (UMAP)',
                  AE='AE')

ds_name = dict(SwissRoll='Swiss Roll',
               Faces='Faces',
               RotatedDigits='Rotated Digits',
               Teapot='Teapot',
               Embryoid='EB Differentiation',
               IPSC='iPSC',
               UMIST='UMIST Faces',
               Tracking='Object Tracking')

metrics_name = dict(dataset='Dataset', model='Model',
                    reconstruction='MSE',
                    R2='$R^2$',
                    rel_reconstruction='Rel. MSE',
                    fit_time='Fit Time (min.)')


# Name getters
def get_model_name(s):
    if s in model_name:
        return model_name[s]
    return s


def get_dataset_name(s):
    if s in ds_name:
        return ds_name[s]
    return s


def get_metrics_name(s):
    if s in metrics_name:
        return metrics_name[s]
    return s
