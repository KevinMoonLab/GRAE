"""Images datasets."""
import os
import urllib

import numpy as np
from torchvision import transforms
import torch
import torchvision.datasets as torch_datasets
from scipy.io import loadmat
from PIL import Image
from scipy import ndimage
import requests
from skimage.transform import resize

from src.data.base import BaseDataset, SEED, FIT_DEFAULT, BASEPATH

ALLOW_CONV = False


class Faces(BaseDataset):
    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED):
        self.url = 'http://stt3795.guywolf.org/Devoirs/D02/face_data.mat'
        self.root = os.path.join(BASEPATH, 'faces')

        if not os.path.exists(self.root):
            os.mkdir(self.root)
            self._download()

        d = loadmat(os.path.join(self.root, 'face_data.mat'))

        x = d['images'].T

        y = d['poses'].T

        super().__init__(x, y, split, split_ratio, seed)

        # Save y_1 and y_2 for coloring
        self.y_2 = self.targets[:, 1].numpy()
        self.y_1 = self.targets[:, 0].numpy()

        # Keep only one latent in the targets attribute for compatibility with
        # other datasets
        self.targets = self.targets[:, 0]

        # Reshape dataset in image format
        if ALLOW_CONV:
            self.data = self.data.view(-1, 1, 64, 64)

    def _download(self):
        print('Downloading Faces dataset...')
        urllib.request.urlretrieve(self.url, os.path.join(self.root, 'face_data.mat'))

    def get_source(self):
        return self.y_1, self.y_2


class Teapot(BaseDataset):
    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED):
        self.url = 'https://github.com/calebralphs/maximum-variance-unfolding/blob/master/Teapots.mat?raw=true'

        self.root = os.path.join(BASEPATH, 'teapot_')

        if not os.path.exists(self.root):
            os.mkdir(self.root)
            self._download()

        d = loadmat(os.path.join(self.root, 'teapot.mat'))

        x = d['Input'][0][0][0].T / 255

        y = np.linspace(start=0, stop=360, num=400, endpoint=False)

        super().__init__(x, y, split, split_ratio, seed)

        self.y = y

        # Reshape dataset in image format
        if ALLOW_CONV:
            def vector_2_rgbimage(vector, h, w, c):
                # Function from https://github.com/calebralphs/maximum-variance-unfolding/blob/master/MVU_Data_Exporation.ipynb
                image_rgb = vector.reshape(c, -1)
                image_rgb = image_rgb.T.reshape(w, h, c)
                image_rgb = np.rot90(image_rgb, 3)

                # Resize for compatibility with Conv architecture
                image_rgb = resize(image_rgb, (76, 128))
                return image_rgb[np.newaxis, :]

            new_data = [vector_2_rgbimage(vec.numpy(), 76, 101, 3) for vec in self.data]
            self.data = torch.from_numpy(np.concatenate(new_data))

            # Test
            # import matplotlib.pyplot as plt
            # plt.imshow(self.data[0])
            # plt.show()

            # Swap dimensions for pytorch conventions
            self.data = self.data.permute(0, 3, 1, 2)

    def _download(self):
        print('Downloading Teapot dataset')
        urllib.request.urlretrieve(self.url, os.path.join(self.root, 'teapot.mat'))

    def get_source(self):
        return self.y


class Tracking(BaseDataset):
    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED):
        self.root = os.path.join(BASEPATH, 'Tracking')

        if not os.path.exists(self.root):
            os.mkdir(self.root)

            # Download sprites
            # All credits go to authors on opengameart : usr_share for sprites and
            # Blarumyrran, Bart, and surt for background
            sprites = Image.open(
                requests.get('https://opengameart.org/sites/default/files/rpg_16x16_0.png', stream=True).raw)
            bg = Image.open(requests.get(
                'https://opengameart.org/sites/default/files/styles/medium/public/tileset_16x16_suurtestbart_1.png',
                stream=True).raw)

            # Crop character and background to needed size
            sprite = sprites.crop((0, 0, 16, 16)).convert('RGBA')

            init_x, init_y = 30, 40
            bg = bg.crop((init_x, init_y, init_x + 64, init_y + 64))

            x = list()
            y_1 = list()
            y_2 = list()

            stride = 1
            i = 0

            while i < 64 - 16:
                j = 0
                while j < 64 - 16:
                    bg_copy = bg.copy()
                    bg_copy.paste(sprite, (i, j), sprite)

                    bg_copy = bg_copy.convert('RGB')

                    # Average for black and white
                    # img = np.mean(np.array(bg_copy), axis=2)
                    img = np.array(bg_copy)

                    # Show samples images
                    # if (i == 3 and j == 3) or (i == 20 and j == 20) or (i == 40 and j == 40):
                    #     Image.fromarray(img).show()

                    x.append(img[np.newaxis, :])
                    y_1.append(i)
                    y_2.append(j)

                    j += stride

                i += stride

            x = np.concatenate(x) / 255
            y_1 = np.array(y_1)
            y_2 = np.array(y_2)

            np.save(os.path.join(self.root, 'x'), x)
            np.save(os.path.join(self.root, 'y'), np.vstack((y_1, y_2)).T)

        # Load data
        x = np.load(os.path.join(self.root, 'x.npy'))
        y = np.load(os.path.join(self.root, 'y.npy'))

        super().__init__(x, y, split, split_ratio, seed)

        if not ALLOW_CONV:
            self.data = self.data.view(len(self), -1)
        else:
            self.data = self.data.permute(0, 3, 1, 2)

        # Save y_1 and y_2 for coloring
        self.y_1 = self.targets[:, 0].numpy()
        self.y_2 = self.targets[:, 1].numpy()

        # Keep only one latent in the targets attribute for compatibility with
        # other datasets
        self.targets = self.targets[:, 0]

    def get_source(self):
        return self.y_1, self.y_2


class Rotated(BaseDataset):
    def __init__(self, fetcher, split='none', split_ratio=FIT_DEFAULT,
                 seed=SEED, n_images=3, n_rotations=360, max_degree=360):
        """Pick n_images of n different classes and return a dataset with
         n_rotations for each image."""
        self.max_degree = max_degree

        np.random.seed(seed)

        transforms_MNIST = transforms.Compose([
            transforms.ToTensor(),
        ])

        train = fetcher(root=BASEPATH, train=True, download=True,
                        transform=transforms_MNIST)

        X = train.data.detach().numpy().reshape(60000, 784)
        X = X / X.max()
        Y = train.targets.detach().numpy()

        # Pick classes
        classes = np.random.choice(10, size=n_images, replace=False)

        imgs = list()

        for i in classes:
            subset = X[Y == i]
            i = np.random.choice(subset.shape[0], size=1)
            imgs.append(subset[i])

        def generate_rotations(img, c, N):
            img = img.reshape((28, 28))

            new_angles = np.linspace(0, self.max_degree, num=N, endpoint=False)

            img_new = np.zeros((N, 28, 28))

            for i, ang in enumerate(new_angles):
                img_new[i, :, :] = ndimage.rotate(img, ang, reshape=False)

                X1 = img_new.reshape(len(new_angles), 784)

            return X1, np.full(shape=(N,), fill_value=c)

        rotations = [generate_rotations(img, c=i, N=n_rotations)
                     for i, img in enumerate(imgs)]

        X_rotated, Y_rotated = zip(*rotations)

        X_rotated = np.concatenate(X_rotated)
        Y_rotated = np.concatenate(Y_rotated)

        super().__init__(X_rotated, Y_rotated, split, split_ratio, seed)


class RotatedDigits(Rotated):
    """3 rotated MNIST digits with 360 rotations per image, for a total of
    1080 samples."""

    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED,
                 n_images=3, n_rotations=360):
        super().__init__(torch_datasets.MNIST, split, split_ratio, seed,
                         n_images, n_rotations)

        if ALLOW_CONV:
            self.data = self.data.view(-1, 1, 28, 28)

            # Test
            # import matplotlib.pyplot as plt
            # plt.imshow(self.data[800][0], cmap='gray')
            # plt.show()
