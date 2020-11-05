import matplotlib.pyplot as plt
import numpy as np

x = np.arange(801)
epoch_max = 800
lam_max = 100

y = (-lam_max * np.exp((x - (epoch_max / 2)) * 0.2)) / (1 + np.exp((x - (epoch_max / 2)) * 0.2))  + lam_max

# self.lam = (-self.epochs * np.exp((epoch - (self.epochs / 2)) * 0.2)) / (1 + np.exp((epoch - (self.epochs / 2)) * 0.2)) \
#            + self.lam_original

plt.plot(x, y)
# plt.xticks([])
# plt.yticks([])
plt.xlabel('Epoch', fontsize=25)
plt.ylabel('λ', fontsize=25)
plt.show()
