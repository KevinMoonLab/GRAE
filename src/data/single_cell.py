"""Single-cell genomics datasets."""
import os

import scprep
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.io import loadmat

from src.data.base import BaseDataset, SEED, FIT_DEFAULT, BASEPATH

EB_COMPONENTS = None  # PCA components for EB data. Set to none if all genes should be kept.


class Embryoid(BaseDataset):
    """Single cell RNA sequencing for embryoid body data generated over 27 day time course."""

    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED):

        # Download and preprocess dataset if needed
        self.root = os.path.join(BASEPATH, 'Embryoid')

        if not os.path.isdir(os.path.join(self.root, "scRNAseq", "T0_1A")):
            os.mkdir(self.root)

            # Code from this section is taken from the PHATE tutorial
            # (https://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb)

            # need to download the data
            scprep.io.download.download_and_extract_zip(
                "https://data.mendeley.com/datasets/v6n743h5ng/1/files/b1865840-e8df-4381-8866-b04d57309e1d/scRNAseq.zip?dl=1",
                self.root)

            # Load data in dataframes
            sparse = True
            T1 = scprep.io.load_10X(os.path.join(self.root, "scRNAseq", "T0_1A"), sparse=sparse, gene_labels='both')
            T2 = scprep.io.load_10X(os.path.join(self.root, "scRNAseq", "T2_3B"), sparse=sparse, gene_labels='both')
            T3 = scprep.io.load_10X(os.path.join(self.root, "scRNAseq", "T4_5C"), sparse=sparse, gene_labels='both')
            T4 = scprep.io.load_10X(os.path.join(self.root, "scRNAseq", "T6_7D"), sparse=sparse, gene_labels='both')
            T5 = scprep.io.load_10X(os.path.join(self.root, "scRNAseq", "T8_9E"), sparse=sparse, gene_labels='both')

            # Filter library size
            filtered_batches = []
            for batch in [T1, T2, T3, T4, T5]:
                batch = scprep.filter.filter_library_size(batch, percentile=20, keep_cells='above')
                batch = scprep.filter.filter_library_size(batch, percentile=75, keep_cells='below')
                filtered_batches.append(batch)
            del T1, T2, T3, T4, T5  # removes objects from memory

            # Merge datasets and create timestamps
            EBT_counts, sample_labels = scprep.utils.combine_batches(
                filtered_batches,
                # ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"],
                np.arange(5),
                append_to_cell_names=True
            )

            # Remove rare genes
            EBT_counts = scprep.filter.filter_rare_genes(EBT_counts, min_cells=10)

            # Normalize
            EBT_counts = scprep.normalize.library_size_normalize(EBT_counts)

            # Remove dead cells
            mito_genes = scprep.select.get_gene_set(EBT_counts,
                                                    starts_with="MT-")  # Get all mitochondrial genes. There are 14, FYI.
            EBT_counts, sample_labels = scprep.filter.filter_gene_set_expression(
                EBT_counts, sample_labels, genes=mito_genes,
                percentile=90, keep_cells='below')

            # Take square root of features
            EBT_counts = scprep.transform.sqrt(EBT_counts)

            np.save(os.path.join(self.root, 'x'), EBT_counts.to_numpy(dtype='float'))
            np.save(os.path.join(self.root, 'y'), sample_labels.to_numpy(dtype='float'))

        file_path = os.path.join(self.root, f'x_{seed}_{EB_COMPONENTS}.npy')  # File path of PCA data

        if EB_COMPONENTS != None and not os.path.exists(file_path):
            # Compute PCA on train split given by a specific seed
            # The naming convention will recompute PCA if the number of components or the seed are changed
            x = np.load(os.path.join(self.root, 'x.npy'), allow_pickle=True)
            n = x.shape[0]
            train_idx, _ = train_test_split(np.arange(n),
                                            train_size=split_ratio,
                                            random_state=seed)

            pca = PCA(n_components=EB_COMPONENTS).fit(x[train_idx])  # Compute only on train set

            np.save(file_path, pca.transform(x))

        if EB_COMPONENTS != None:
            # Load pca data
            x = np.load(file_path, allow_pickle=True)
        else:
            # Load raw dataset
            x = np.load(os.path.join(self.root, 'x.npy'), allow_pickle=True)

        y = np.load(os.path.join(self.root, 'y.npy'), allow_pickle=True)

        super().__init__(x, y, split, split_ratio, seed)

    def get_source(self):
        # Only 1D ground truth
        return self.targets.numpy(), None


class IPSC(BaseDataset):
    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED, subsample=100000):
        file_path = os.path.realpath(os.path.join(BASEPATH, 'Ipsc', 'ipscData.mat'))
        if not os.path.exists(file_path):
            raise Exception(f'{file_path} should be added manually before running experiments.')

        data = loadmat(file_path)
        x = data['data']
        y = data['data_time']

        if subsample is not None:
            np.random.seed(seed)
            mask = np.random.choice(x.shape[0], size=subsample)
            x = x[mask]
            y = y[mask]

        super().__init__(x, y, split, split_ratio, seed)
