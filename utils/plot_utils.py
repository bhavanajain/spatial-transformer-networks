import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from math import ceil

def demo_simple_grid(images, figname=None, cmap='gray', axes_pad=0.05):
    grid_size = int(ceil(images.shape[0] ** 0.5))
    fig = plt.figure()
    grid = AxesGrid(fig, 111, nrows_ncols = (grid_size, grid_size), axes_pad = axes_pad)
    for i in range(images.shape[0]):
        im = grid[i].imshow(images[i], cmap=cmap)
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
    plt.close("all")
