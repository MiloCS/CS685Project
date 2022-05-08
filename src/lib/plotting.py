import numpy as np
import matplotlib.pyplot as plt


def hexbin_plot(x, y, labels):
    types = set(labels)

    n = len(types)
    sqrt_n = np.sqrt(n)
    if sqrt_n % 1 == 0:
        r, c = sqrt_n, sqrt_n
    else:
        r, c = int(sqrt_n) + 1, int(sqrt_n)

    fig, axes = plt.subplots(r,c, figsize=(c*3, r*3))

    axes = axes.flatten()
    for i, label in enumerate(types):
        inds = np.where(labels == label)[0]
        axes[i].hexbin(x[inds], y[inds], gridsize=20)
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])