"""Disentanglement metrics."""
import numpy as np
from skbio.stats.distance import mantel
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist
from sklearn.decomposition import FastICA

SEED = 42
def MI(source, z, random_state=SEED):
    # Thin wrapper for compatibility with pearonr and spearmanr
    z = z.reshape((-1, 1))
    return mutual_info_regression(z, source,
            discrete_features=False, random_state=random_state)[0], None



def cuts(source_ref, source_corr, z, corr, **kwargs):
    # Partition data over equal-width intervals of source_ref, then apply
    # similarity measure corr to source_corr and z over said partitions.
    # Return the average
    n = 10
    slices = np.linspace(source_ref.min(), source_ref.max(), endpoint=True, num=n + 1)
    slices[-1] += .1  # Increase last boundary to include last point

    total = 0

    for i in range(n):
        mask = np.logical_and(source_ref >= slices[i],
                              source_ref < slices[i + 1])
        source_corr_m, z_m = source_corr[mask], z[mask]
        c, _ = corr(source_corr_m, z_m, **kwargs)
        total += abs(c)

    return total / n


def slice_correlator(source_1, source_2, z_1, z_2, corr, name, **kwargs):
    # Check with optimal corr if flipping the axes is required
    flip_required = optimal_corr(source_1, source_2, z_1, z_2,
                                 corr, name, check_flip=True, **kwargs)
    if flip_required:
        z_temp = z_2
        z_2 = z_1
        z_1 = z_temp

    key_1 = f'{name}_slice_source_1'
    key_2 = f'{name}_slice_source_2'

    # Compute similariy measure over 20 sections (10 sections orthogonal to
    # each axes)
    return {
        key_1: cuts(source_2, source_1, z_1, corr, **kwargs),
        key_2: cuts(source_1, source_2, z_2, corr, **kwargs),
    }



def optimal_corr(source_1, source_2, z_1, z_2, corr, name, check_flip=False, **kwargs):
    # Correlate a 2D embedding with two source signals with optimal matching and
    # sign flipping

    # First candidate
    a_1, _ = corr(source_1, z_1, **kwargs)
    a_2, _ = corr(source_2, z_2, **kwargs)
    a_1 = abs(a_1)
    a_2 = abs(a_2)
    s_1 = a_1 + a_2

    # Second candidate
    b_1, _ = corr(source_1, z_2, **kwargs)
    b_2, _ = corr(source_2, z_1, **kwargs)
    b_1 = abs(b_1)
    b_2 = abs(b_2)
    s_2 = b_1 + b_2

    # Return matching that maximizes correlations over both axes
    key_1 = f'{name}_source_1'
    key_2 = f'{name}_source_2'

    if check_flip:
        # Simply return boolean indicating if flipping is required. Used by
        # slice_correlator
        if s_1 > s_2:
            return False
        else:
            return True

    if s_1 > s_2:
        return {key_1: a_1, key_2: a_2}
    else:
        return {key_1: b_1, key_2: b_2}


def optimal_corr_ICA(source_1, source_2, z_1, z_2, corr, name, random_state=SEED, **kwargs):
    # Optimal correlation with FastICA preprocessing
    z = np.vstack((z_1, z_2)).T
    z_transformed = FastICA(random_state=random_state).fit_transform(z)

    return optimal_corr(source_1, source_2, *z_transformed.T, corr, name + '_ICA', **kwargs )


def dis_metrics(source_1, source_2, z_1, z_2):
    # Return three variants of Pearson, Spearman and MI, as well as Mantel
    metrics = dict()

    for f, name in ((pearsonr, 'pearson'), (spearmanr, 'spearman'), (MI, 'mutual_information')):
        args = {}

        corr = optimal_corr(source_1, source_2, z_1, z_2, f, name, **args)
        corr_ICA = optimal_corr_ICA(source_1, source_2, z_1, z_2, f, name, **args)
        corr_slice = slice_correlator(source_1, source_2, z_1, z_2, f, name, **args)
        metrics.update({**corr, **corr_ICA, **corr_slice})

    # Mantel
    source = np.vstack((source_1, source_2)).T
    z = np.vstack((z_1, z_2)).T
    source_dist = pdist(source, metric='euclidean')
    z_dist = pdist(z, metric='euclidean')
    dist_corr, _, _ = mantel(source_dist, z_dist, permutations=1)

    return {**metrics, 'dist_corr': dist_corr}