import sys
import time
import math

import numpy as np
from matplotlib import pyplot as plt

from numba import jit

from __init__ import setup, mask, data, shape


def thresholded_dispersion(image, mask, sigma_s=3):
    """Return a boolean map the same shape as image, mask which contains 1 for
    signal pixels, 0 for background / masked."""

    # mask out the bad pixels
    image = image * mask

    # compute mean, variance, dispersion (all in one loop) and save threshold
    # map at the end...
    signal = np.zeros(shape=image.shape, dtype=np.uint8)

    # to keep things safe type-wise (i.e. range of values) do one module
    # at a time i.e. in 32 chunks...
    ny, nx = image.shape
    ny = ny // 32

    # compute integral images from one module at a time
    for module in range(32):

        # grab the data
        _image = image[module * ny : module * ny + ny, :].astype(np.uint32)
        _mask = mask[module * ny : module * ny + ny, :].astype(np.uint32)

        # compute interal images
        isum = _image.cumsum(axis=0).cumsum(axis=1)
        isum2 = (_image ** 2).cumsum(axis=0).cumsum(axis=1)
        nsum = _mask.cumsum(axis=0).cumsum(axis=1)

        # compute dispersion - must be able to find less dumb way than this
        for i in range(3, ny - 3):
            for j in range(3, nx - 3):
                n = (
                    nsum[i + 3, j + 3]
                    + nsum[i - 3, j - 3]
                    - nsum[i - 3, j + 3]
                    - nsum[i + 3, j - 3]
                )
                I = (
                    isum[i + 3, j + 3]
                    + isum[i - 3, j - 3]
                    - isum[i - 3, j + 3]
                    - isum[i + 3, j - 3]
                )
                I2 = (
                    isum2[i + 3, j + 3]
                    + isum2[i - 3, j - 3]
                    - isum2[i - 3, j + 3]
                    - isum2[i + 3, j - 3]
                )
                if n > 1 and I > 0:
                    mean = I / n
                    variance = (I2 - I ** 2 / n) / n
                    if (variance / mean) > 1 + sigma_s * math.sqrt(2 / (n - 1)):
                        signal[module * ny + i, j] = 1

    return signal


def simple():
    filename = sys.argv[1]
    setup(filename)

    m = mask()

    nz, ny, nx = shape()

    d = data(0)

    t0 = time.time()
    signal = thresholded_dispersion(d, m)
    t1 = time.time()

    plt.imshow(signal)
    plt.show()


simple()
