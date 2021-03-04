"""PHATE, UMAP and procrustes tools."""
import numpy as np
import phate
import umap
from sklearn.decomposition import PCA as SKPCA
from sklearn.pipeline import make_pipeline

from grae.models.base_model import BaseModel, SEED
from grae.models.external_tools.procrustes import procrustes

PROC_THRESHOLD = 20000
PROC_BATCH_SIZE = 5000
PROC_LM = 1000


def fit_transform_procrustes(x, fit_transform_call, procrustes_batch_size, procrustes_lm):
    """Fit model and transform data for larger datasets.

    If dataset has more than self.proc_threshold samples, then compute PHATE over
    mini-batches. In each batch, add self.procrustes_lm samples (which are the same for all batches),
    which can be used to compute a  procrustes transform to roughly align all batches in a coherent manner.
    This last step is required since PHATE can lead to embeddings with different rotations or reflections
    depending on the batch.

    Args:
        x(BaseDataset): Dataset to fit and transform.
        fit_transform_call(callback): fit & transform method of an sklearn-style estimator.
        procrustes_batch_size(int): Batch size of procrustes approach.
        procrustes_lm (int): Number of anchor points present in all batches. Used as a reference for the procrustes
        transform.

    Returns:
        ndarray: Embedding of x, which is the union of all batches aligned with procrustes.

    """
    lm_points = x[:procrustes_lm, :]  # Reference points included in all batches
    initial_embedding = fit_transform_call(lm_points)
    result = [initial_embedding]
    remaining_x = x[procrustes_lm:, :]
    while len(remaining_x) != 0:
        if len(remaining_x) >= procrustes_batch_size:
            new_points = remaining_x[:procrustes_batch_size, :]
            remaining_x = np.delete(remaining_x,
                                    np.arange(procrustes_batch_size),
                                    axis=0)
        else:
            new_points = remaining_x
            remaining_x = np.delete(remaining_x,
                                    np.arange(len(remaining_x)),
                                    axis=0)

        subsetx = np.vstack((lm_points, new_points))
        subset_embedding = fit_transform_call(subsetx)

        d, Z, tform = procrustes(initial_embedding,
                                 subset_embedding[:procrustes_lm, :])

        subset_embedding_transformed = np.dot(
            subset_embedding[procrustes_lm:, :],
            tform['rotation']) + tform['translation']

        result.append(subset_embedding_transformed)
    return np.vstack(result)


class PHATE(phate.PHATE, BaseModel):
    """Wrapper for PHATE to work with torch datasets.

    Also add procrustes transform when dealing with large datasets for improved scalability.
    """

    def __init__(self, proc_threshold=PROC_THRESHOLD, procrustes_batches_size=PROC_BATCH_SIZE,
                 procrustes_lm=PROC_LM, **kwargs):
        """Init.

        Args:
            proc_threshold(int): Threshold beyond which PHATE is computed over mini-batches of the data and batches are
            realigned with procrustes. Otherwise, vanilla PHATE is used.
            procrustes_batches_size(int): Batch size of procrustes approach.
            procrustes_lm (int): Number of anchor points present in all batches. Used as a reference for the procrustes
            transform.
            **kwargs: Any remaining keyword arguments are passed to the PHATE model.
        """
        self.proc_threshold = proc_threshold
        self.procrustes_batch_size = procrustes_batches_size
        self.procrustes_lm = procrustes_lm
        self.comet_exp = None
        super().__init__(**kwargs)

    def fit_transform(self, x):
        """Fit model and transform data.

        Overrides PHATE fit_transform method on datasets larger than self.proc_threshold to compute PHATE over
        mini-batches with procrustes realignment.

        Args:
            x(BaseDataset): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x.

        """
        x, _ = x.numpy()

        if x.shape[0] < self.proc_threshold:
            result = super().fit_transform(x)
        else:
            print('            Fitting procrustes...')
            result = self.fit_transform_procrustes(x)
        return result

    def fit_transform_procrustes(self, x):
        """Fit model and transform data for larger datasets.

        If dataset has more than self.proc_threshold samples, then compute PHATE over
        mini-batches. In each batch, add self.procrustes_lm samples (which are the same for all batches),
        which can be used to compute a  procrustes transform to roughly align all batches in a coherent manner.
        This last step is required since PHATE can lead to embeddings with different rotations or reflections
        depending on the batch.

        Args:
            x(BaseDataset): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x, which is the union of all batches aligned with procrustes.

        """
        return fit_transform_procrustes(x, super().fit_transform, self.procrustes_batch_size, self.procrustes_lm)


class UMAP(BaseModel):
    """Wrapper for UMAP to work with torch datasets and the procrustes approach described in the paper.

    """

    def __init__(self, random_state=SEED, proc_threshold=PROC_THRESHOLD, procrustes_batch_size=PROC_BATCH_SIZE,
                 procrustes_lm=PROC_LM, **kwargs):
        """Init.

        Args:
            random_state(int): For seeding.
            proc_threshold(int): Threshold beyond which PHATE is computed over mini-batches of the data and batches are
            realigned with procrustes. Otherwise, vanilla PHATE is used.
            procrustes_batches_size(int): Batch size of procrustes approach.
            procrustes_lm (int): Number of anchor points present in all batches. Used as a reference for the procrustes
            transform.
            **kwargs: Any remaining keyword arguments are passed to the UMAP estimator.
        """
        super().__init__()
        self.umap_estimator = umap.UMAP(random_state=random_state, **kwargs)
        self.estimator = None
        self.data_shape = None
        self.proc_threshold = proc_threshold
        self.procrustes_batch_size = procrustes_batch_size
        self.procrustes_lm = procrustes_lm

    def init_estimator(self, x):
        """Init estimator by adding a PCA step if need be.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        steps = [self.umap_estimator]
        if x.shape[1] > 100 and x.shape[0] > 1000:
            # Note : PHATE does a PCA step by default. See their doc.
            print('More than 100 dimensions and 1000 samples. Adding PCA step to UMAP pipeline.')
            steps = [SKPCA(n_components=100)] + steps

        self.estimator = make_pipeline(*steps)

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        x, _ = x.numpy()
        self.init_estimator(x)
        self.estimator.fit(x)

    def fit_transform(self, x):
        """Fit model and transform data.

        Args:
            x(BaseDataset): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x.

        """
        if len(x) < self.proc_threshold:
            self.fit(x)
            result = self.transform(x)
        else:
            print('            Fitting procrustes...')
            result = self.fit_transform_procrustes(x)
        return result

    def transform(self, x):
        """Transform new data to the low dimensional space.

        Args:
            x(BaseDataset): Dataset to transform.
        Returns:
            ndarray: Embedding of x.

        """
        x, _ = x.numpy()

        # To prevent overwriting original data. Related to issue #515 of the UMAP repo?
        # Preserving _raw_data ensures the fitted estimator can transform multiple times different datasets (i.e. train
        # test) from the same state. The attribute is otherwise altered and future calls to transform will lead
        # to poor results. Does not affect test metrics.
        raw_data_bk = self.estimator[-1]._raw_data.copy()
        results = self.estimator.transform(x)
        self.estimator[-1]._raw_data = raw_data_bk
        return results

    def fit_transform_procrustes(self, x):
        """See comments for the equivalent PHATE method.

        Args:
            x(BaseDataset): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x, which is the union of all batches aligned with procrustes.

        """
        x, _ = x.numpy()
        self.init_estimator(x)
        return fit_transform_procrustes(x, self.estimator.fit_transform,
                                        self.procrustes_batch_size, self.procrustes_lm)

    def reconstruct(self, x):
        """Transform and inverse x.

        Args:
            x(BaseDataset): Data to transform and reconstruct.

        Returns:
            ndarray: Reconstructions of x.

        """
        data_len = len(x)
        data_shape = x[0][0].shape
        return self.estimator.inverse_transform(self.transform(x)).reshape((data_len, *data_shape))

    def inverse_transform(self, x):
        """Take coordinates in the embedding space and invert them to the data space.

        Args:
            x(ndarray): Points in the embedded space with samples on the first axis.
        Returns:
            ndarray: Inverse (reconstruction) of x.

        """
        # To prevent overwriting original data. Related to issue #515 of the UMAP repo?
        # See comments under the transform method
        raw_data_bk = self.estimator[-1]._raw_data.copy()
        results = self.estimator.inverse_transform(x)
        self.estimator[-1]._raw_data = raw_data_bk
        return results


class PCA(BaseModel):
    """Wrapper for PCA to work with torch datasets. Inherit utility methods from BaseModel."""

    def __init__(self, n_components=2, **kwargs):
        """Init.

        Args:
            n_components(int): Number of principal components to keep.
            **kwargs: Any remaining keyword arguments are passed to the sklearn PCA class.
        """
        self.comet_exp = None
        self.estimator = SKPCA(n_components=n_components, **kwargs)

    def fit_transform(self, x):
        """Fit model and transform data.

        Args:
            x(BaseDataset): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x.

        """
        x, _ = x.numpy()
        return self.estimator.fit_transform(x)

    def fit(self, x):
        """Fit data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        x, _ = x.numpy()
        self.estimator.fit(x)

    def transform(self, x):
        """Transform data.

        Args:
            x(BaseDataset): Dataset to transform.

        Returns:
            ndarray: Embedding of x.

        """
        x, _ = x.numpy()
        return self.estimator.transform(x)

    def inverse_transform(self, x):
        """Inverse data back to input space.

        Args:
            x(ndarray): Dataset to inverse.

        Returns:
            ndarray: Inverse of x.

        """
        return self.estimator.inverse_transform(x)

    def reconstruct(self, x):
        """Transform and inverse x.

        Args:
            x(BaseDataset): Data to transform and reconstruct.

        Returns:
            ndarray: Reconstructions of x.

        """
        data_len = len(x)
        data_shape = x[0][0].shape
        return self.estimator.inverse_transform(self.transform(x)).reshape((data_len, *data_shape))
