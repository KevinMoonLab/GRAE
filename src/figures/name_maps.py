"""Prettier names for models, datasets and metrics."""
model_name = dict(Umap_t='UMAP',
                  Umap='UMAP',
                  diffusion_net='Diffusion Nets',
                  PHATE='PHATE',
                  TSNE='t-SNE',
                  AE='Autoencoder')

base_name = dict(GRAE='GRAE (',
                 GRAE_TSNE='GRAE t-SNE (',
                 GRAE_UMAP='GRAE UMAP (',
                 TopoAE='TAE (',
                 SGRAE='Siamese GRAE (',
                 SoftGRAE='Soft GRAE (')

ds_name = dict(SwissRoll='Swiss Roll',
               Faces='Faces',
               RotatedDigits='Rotated Digits',
               Teapot='Teapot',
               Embryoid='EB Differentiation',
               IPSC='IPSC',
               UMIST='UMIST Faces',
               Tracking='Object Tracking')

metrics_name = dict(dataset='Dataset', model='Model',
                    continuity='Cont', trustworthiness='Trust', mrre='MRRE',
                    reconstruction='MSE',
                    R2='$R^2$',
                    rel_reconstruction='Rel. MSE',
                    corr_source='Correlation',
                    corr_source_ICA='Correlation ICA',
                    pearson='Pearson',
                    spearman='Spearman',
                    mutual_information='MI',
                    pearson_ICA='Pearson ICA',
                    spearman_ICA='Spearman ICA',
                    mutual_information_ICA='MI ICA',
                    pearson_slice='Pearson Section',
                    spearman_slice='Spearman Section',
                    mutual_information_slice='MI Section',
                    dist_corr='Distance Corr.',
                    corr_source_patch='Patch Correlation',
                    corr_source_ICA_patch='Patch Correlation ICA',
                    )

for i in (5, 10, 20):
    for s in ('continuity', 'trustworthiness', 'mrre'):
        base = metrics_name[s]
        metrics_name[f'{s}_{i}'] = f'{base} ({i})'


def get_model_name(m):
    if m not in model_name:
        splits = m.split('_')
        base, lam = splits[0], splits[1]
        return base_name[base] + f'{lam})'

    return model_name[m]


def get_dataset_name(m):
    return ds_name[m]


def get_metrics_name(m):
    return metrics_name[m]
