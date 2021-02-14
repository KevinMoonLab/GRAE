"""Routine to score embeddings."""
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr


class PolarConverter:
    """Pipeline object to convert x to polar coordinates. Only returns the angle."""

    def fit(self, x, y):
        """Fit method.

        Align one angle with the ground truth (by rotating the embedding) to prevent arbitrary rotations
        from affecting downstream regression tasks.

        Args:
            x(ndarray): Input.
            y(ndarray): Ground truth angles.
        """
        x = x.copy()

        self.mean = x.mean(axis=0)
        self.y = y

        # Center embedding
        x -= self.mean

        # Polar coordinates of the embedding (we're only interested in the angle here)
        phi = np.arctan2(*x.T) + np.pi

        # Align smallest angle to get a rough alignment and prevent arbitrary rotations from affecting
        # downstream regression tasks
        arg_min = y.argmin()
        min_offset = phi[arg_min] - y[arg_min]
        phi -= min_offset
        phi %= 2 * np.pi

        # Further adjust offset to maximize correlation with labels
        search_range = min(x.shape[0] - x.shape[0] % 2, 10)
        sort_idx = phi.argsort()
        corr = list()

        for i in range(2 * search_range):
            new_phi = phi - phi[sort_idx[i - search_range]]
            new_phi %= 2 * np.pi
            candidate_corr, _ = pearsonr(new_phi, self.y)
            corr.append(candidate_corr)

        corr = np.abs(np.array(corr))
        self.phi_offset = min_offset + phi[sort_idx[corr.argmax() - search_range]]

        return self

    def transform(self, x):
        """Fit method.
        Args:
            x(ndarray): Input.
        """
        x = x.copy()
        x -= self.mean
        phi = np.arctan2(*x.T) + np.pi
        phi -= self.phi_offset
        phi %= 2 * np.pi

        return phi.reshape((-1, 1))


class EmbeddingProber:
    """Class to benchmark MSE, the coefficient of determination (R2) for ground truth continuous variables and
    classification accuracy of dataset has labels.
    """
    def fit(self, model, dataset, mse_only=False):
        """Fit regressors to predict latent variables and/or labels if available.

        If a dataset has multiple latent variables, one regressor is used per variable. Moreover, if the dataset has
        latent variables in addition to class labels, the data is divided according to labels and one regressor
        is trained per combination of label/latent variable.

        Args:
            model(BaseModel): Fitted Model.
            dataset(BaseDataset): Dataset to benchmark.
            mse_only(optional, bool): Compute only MSE. Useful for lightweight computations during hyperparameter search.

        """
        self.linear_regressors = []
        self.linear_classifiers = []
        self.mse_only = mse_only

        # Get data embedding and train MSE metrics
        self.model = model
        self.z_train, self.rec_train_metrics = model.score(dataset)
        n_components = self.z_train.shape[1]

        # Fit regressions (one per combination of class and latent variable)
        if dataset.latents is not None and not mse_only:
            # Use dummy labels for one class if no class labels are provided or if partition mode is turned off
            if dataset.labels is None or not dataset.partition:
                labels = np.zeros(len(dataset))
            else:
                labels = dataset.labels[:, 0]

            c = np.unique(labels)

            # Check if classes are correctly indexed
            for i in range(len(c)):
                if i != c[i]:
                    raise ValueError('Class labels should be indexed from 0 to no_of_classes - 1.')

            for i in c:
                mask = labels == i
                z_c = self.z_train[mask]
                y_c = dataset.latents[mask]
                self.linear_regressors.append([])

                for j, latent in enumerate(y_c.T):
                    scaler = PolarConverter() if (j in dataset.is_radial and n_components == 2) else StandardScaler()
                    pipeline = Pipeline(steps=[('scaler', scaler),
                                               ('regression', SGDRegressor())])
                    pipeline.fit(z_c, latent)
                    self.linear_regressors[int(i)].append(pipeline)

        # Fit one linear classifier per class of labels
        if dataset.labels is not None and not mse_only:
            for i, classes in enumerate(dataset.labels.T):
                pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('regression',
                                                                          LogisticRegression(max_iter=1000))])
                pipeline.fit(self.z_train, classes)
                self.linear_classifiers.append(pipeline)

    def score(self, dataset, is_train=False):
        """Score dataset.

        Args:
            dataset(BaseDataset): Dataset to score.
            is_train(optional, bool): If True, will reuse embeddings and MSE metrics computed during fit.

        Returns:
            (tuple) tuple containing:
                z(ndarray): Data embedding.
                metrics(dict[float]): Dict of metrics.

        """
        metrics = dict()

        if is_train:
            z, rec_metrics = self.z_train, self.rec_train_metrics
        else:
            z, rec_metrics = self.model.score(dataset)

        metrics.update(rec_metrics)

        # Fit regressions (one per combination of class and latent variable)
        if dataset.latents is not None and not self.mse_only:
            r2 = list()

            # Use dummy labels for one class if no class labels are provided or if partition mode is turned off
            if dataset.labels is None or not dataset.partition:
                labels = np.zeros(len(dataset))
            else:
                labels = dataset.labels[:, 0]

            c = np.unique(labels)

            # Check if classes are correctly indexed
            for i in range(len(c)):
                if i != c[i]:
                    raise ValueError('Class labels should be indexed from 0 to no_of_classes - 1.')

            for i in c:
                mask = labels == i
                z_c = z[mask]
                y_c = dataset.latents[mask]

                for j, latent in enumerate(y_c.T):
                    r2.append(self.linear_regressors[int(i)][j].score(z_c, latent))

            metrics.update({'R2': np.mean(r2)})
        else:
            metrics.update({'R2': -1})

        # Fit one linear classifier per class of labels
        if dataset.labels is not None and not self.mse_only:
            acc = list()
            for i, classes in enumerate(dataset.labels.T):
                acc.append(self.linear_classifiers[int(i)].score(z, classes))

            metrics.update({'Acc': np.mean(acc)})
        else:
            metrics.update({'Acc': -1})

        return z, metrics
