from src.data import Teapot

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(46)


data = Teapot(split='none')

fig, axs = plt.subplots(2, 4, figsize=(4 * 5.5, 2 * 3.5))

for ax in axs.reshape(-1):
    i = np.random.choice(len(data), size=1)
    img = data[i][0][0]
    ax.imshow(img.permute(1, 2, 0))
    ax.set_xticks([])
    ax.set_yticks([])


plt.show()