"""Images datasets."""
import os
import urllib
import math

import numpy as np
from torchvision import transforms
import torch
import torchvision.datasets as torch_datasets
from scipy.io import loadmat
from PIL import Image
from scipy import ndimage
import requests
from skimage.transform import resize
from skimage.util import random_noise
import zipfile

from src.data.base import BaseDataset, SEED, FIT_DEFAULT, BASEPATH

ALLOW_CONV = True


class Faces(BaseDataset):
    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED):
        self.url = 'http://stt3795.guywolf.org/Devoirs/D02/face_data.mat'
        self.root = os.path.join(BASEPATH, 'Faces')

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

    def get_latents(self):
        return np.vstack((self.y_1.flatten(), self.y_2.flatten())).T


class UMIST(BaseDataset):
    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED):
        self.url = 'https://cs.nyu.edu/~roweis/data/umist_cropped.mat'
        self.root = os.path.join(BASEPATH, 'UMIST')

        if not os.path.exists(self.root):
            os.mkdir(self.root)
            self._download()

        d = loadmat(os.path.join(self.root, 'UMIST.mat'))

        data = d['facedat'][0]
        x = list()
        targets = list()

        for i, p in enumerate(data):
            p = np.moveaxis(p, (0, 1), (1, 2))

            p.reshape((-1, 112 * 92)) # Flat array

            # Some subjects require reordering for order to roughly match rotation angles

            # Person 6
            if i == 6:
                mask = [0, 1, 2, 18, 17, 16, 15, 14, 13, 12, 3, 11, 10, 4, 9, 5, 6, 8, 7]
                p = p[mask]

            # Person 8
            if i == 8:
                mask = [0, 1, 2, 3, 19, 18, 17, 4, 16, 5, 6, 15, 7, 14, 8, 9, 10, 11, 13, 12]
                p = p[mask]

            # 9_30 is an outlier

            # Person 13
            if i == 13:
                mask = [0, 14, 15, 16, 17, 18, 1, 13, 19, 20, 2, 21, 3, 4, 22, 12,
                        23, 5, 24, 6, 25, 26, 11, 7, 27, 28, 8, 29, 9, 10]
                p = p[mask]

            # The last 3-4 images of person 14 have different head angles
            # Person 15
            if i == 15:
                mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 18, 19, 20, 21, 22]
                p = p[mask]

            # Person 19
            if i == 19:
                mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        28, 29, 30, 31, 32, 23, 27, 33, 24, 25, 26]
                p = p[mask]

            x.append(p)
            targets.append(np.vstack((i * np.ones(p.shape[0]), np.arange(p.shape[0]))).T)

        x = np.vstack(x)/255
        targets = np.vstack(targets)

        super().__init__(x, targets, split, split_ratio, seed, labels=targets[:, 0].astype(int).reshape(-1))

        # Store both classes and angles as latents
        self.latents = np.copy(self.targets)

        # Keep only classes as targets
        self.targets = self.targets[:, 0]

        # Reshape dataset in image format
        if ALLOW_CONV:
            self.data = self.data.view(-1, 1, 112, 92)

    def _download(self):
        print('Downloading UMIST dataset...')
        urllib.request.urlretrieve(self.url, os.path.join(self.root, 'UMIST.mat'))

    def get_latents(self):
        return self.latents

class Teapot(BaseDataset):
    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED):
        self.url = 'https://github.com/calebralphs/maximum-variance-unfolding/blob/master/Teapots.mat?raw=true'

        self.root = os.path.join(BASEPATH, 'Teapot')

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

    def get_latents(self):
        """Return vector of 1 for compatibility with other rotations datasets (even if only 1 class here)."""
        return np.vstack((np.ones(len(self)), self.targets.numpy().flatten() / (360/(2 * math.pi)))).T


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

                    # Add noise using skimage
                    bg_copy = np.array(bg_copy)
                    bg_copy = random_noise(bg_copy, mode='gaussian',
                                           seed=((43 * i + 1) * (101 * j + 1)) % 74501, var=0.05)
                    bg_copy = Image.fromarray((bg_copy * 255).astype(np.uint8))

                    bg_copy.paste(sprite, (i, j), sprite)

                    bg_copy = bg_copy.convert('RGB')

                    # Average for black and white
                    # img = np.mean(np.array(bg_copy), axis=2)
                    img = np.array(bg_copy)

                    # Show samples images
                    if (i == 3 and j == 3) or (i == 20 and j == 20) or (i == 40 and j == 40):
                        Image.fromarray(img).show()

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

    def get_latents(self):
        return np.vstack((self.y_1, self.y_2)).T


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

            return X1, np.full(shape=(N,), fill_value=c), new_angles

        rotations = [generate_rotations(img, c=i, N=n_rotations)
                     for i, img in enumerate(imgs)]

        inputs, targets, angles = zip(*rotations)

        inputs = np.concatenate(inputs)
        targets = np.concatenate(targets)
        angles = np.concatenate(angles)

        # Send targets and angles as one object for compatibility with parent class
        targets = np.vstack((targets, angles)).T

        super().__init__(inputs, targets, split, split_ratio, seed)

        # Split back angles and targets
        self.angles = self.targets[:, 1]
        self.targets = self.targets[:, 0]

    def get_latents(self):
        """Class as first column and angles (in radians) as second column."""
        return np.vstack((self.targets.numpy().flatten(), self.angles.numpy().flatten() / (360/(2 * math.pi)))).T


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


class COIL100(BaseDataset):
    def __init__(self, split='none', split_ratio=FIT_DEFAULT, seed=SEED):
        self.url = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip'
        self.root = os.path.join(BASEPATH, 'COIL100')

        if not os.path.exists(self.root):
            os.mkdir(self.root)
            self._download()

        x = np.load(os.path.join(self.root, 'x.npy'))
        y = np.load(os.path.join(self.root, 'y.npy'))

        super().__init__(x, y, split, split_ratio, seed, labels=y[:, 0].astype(int).reshape(-1))

        self.latents = np.copy(self.targets)

        # Keep only one latent in the targets attribute for compatibility with
        # other datasets
        self.targets = self.targets[:, 0]

        # Reshape dataset in image format
        if ALLOW_CONV:
            self.data = self.data.view(-1, 128, 128, 3).permute(0, 3, 1, 2)

    def _download(self):
        print('Downloading COIL100 dataset...')
        path = os.path.join(self.root, 'coil100.zip')
        urllib.request.urlretrieve(self.url, path)

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall()

        x = list()
        y = list()

        for i in range(1, 101):
            for j in range(0, 360, 5):
                img = Image.open(os.path.join(self.root, 'coil-100', f'obj{i}__{j}.png'))
                img.load()
                img = np.asarray(img, dtype='int32')
                x.append(img.flatten()/255)
                y.append(np.array([i, j]))

        x = np.vstack(x)
        y = np.vstack(y)
        np.save(os.path.join(self.root, 'x.npy'), x)
        np.save(os.path.join(self.root, 'y.npy'), y)


    def get_latents(self):
        return self.latents
