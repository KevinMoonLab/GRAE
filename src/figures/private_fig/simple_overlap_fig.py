import numpy as np
import matplotlib.pyplot as plt


def manifold(domain):
    return np.vstack((domain, np.abs(domain))).T


def fone(x):
    return np.vstack((x[:, 0], np.zeros(x.shape[0]))).T


def fone_inv(z):
    z = z[:, 0]
    return np.vstack((z, np.abs(z))).T


def ftwo(x):
    return np.vstack((x[:, 1], np.zeros(x.shape[0]))).T


def ftwo_inv(z):
    z = z[:, 0]
    return np.vstack((z * np.where((z+.5) % 2 < 1, -1, 1), z)).T


landmarks = np.array([
    [-6, 6],
    [-4, 4],
    [-2, 2],
    [0, 0],
    [1, 1],
    [3, 3],
    [5, 5]
])
colors = ['g', 'g', 'g', 'blue', 'orange', 'orange', 'orange']
# colors = np.arange(7)
# map = 'magma'
fig, ax = plt.subplots(2, 3, figsize=(3 * 3.5, 2 * 3.5))

grid = [(fone, fone_inv), (ftwo, ftwo_inv)]

for i, (f, inv) in enumerate(grid):
    domain = np.linspace(-6.45, 6.5, 1000, endpoint=False)
    m = manifold(domain)
    ax[i, 0].scatter(*m.T, s=1)
    ax[i, 0].scatter(*landmarks.T, s=100, c=colors)
    # ax[i, 1].set_yticks([])
    ax[i, 0].set_xlim((-7.5, 7.5))
    ax[i, 0].set_ylim((-1, 7))


    z = f(m)
    z_l = f(landmarks)
    if i == 0:
        ax[i, 1].scatter(*z.T, s=1)
        ax[i, 1].scatter(*z_l.T, s=100,  c=colors)
        ax[i, 1].set_yticks([])
    else:
        ax[i, 1].scatter(*np.flip(z).T, s=1)
        ax[i, 1].scatter(*np.flip(z_l).T, s=100,  c=np.flip(colors))
        ax[i, 1].set_xticks([])

    ax[i, 1].set_xlim((-7.5, 7.5 ))
    ax[i, 1].set_ylim((-1, 7))

    m_prime = inv(z)
    l_prime = inv(z_l)
    ax[i, 2].scatter(*m_prime.T, s=1)
    ax[i, 2].scatter(*l_prime.T, s=100, c=colors)
    ax[i, 2].set_xlim((-7.5, 7.5))
    ax[i, 2].set_ylim((-1, 7))
    # ax[i, 2].set_yticks([])


ax[0, 0].set_title('Manifold')
# ax[0, 0].set_xticks([])
# ax[0, 2].set_xticks([])
ax[0, 1].set_title('1D Latent space')
ax[0, 2].set_title('Reconstruction')
plt.show()
