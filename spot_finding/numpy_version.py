import sys
import time
import math

import numpy as np
from matplotlib import pyplot as plt

from numba import jit

from spot_finder_data import setup, mask, data, shape, rettilb


def kernel_summation(data, knl=7):
    """Return the summed SAT such that reading off out[i,j] gives you the
    sum of data in a kernel box around i,j. Assumes data were already padded
    by (kernel - 1) / 2 pixels."""

    s = data.cumsum(axis=0).cumsum(axis=1)
    return s[knl:, knl:] + s[:-knl, :-knl] - s[:-knl, knl:] - s[knl:, :-knl]


def thresholded_dispersion(image, mask, sigma_s=3, knl=7):
    """Return a boolean map the same shape as image, mask which contains 1 for
    signal pixels, 0 for background / masked."""

    pad = (knl - 1) // 2

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
        _image = np.pad(
            image[module * ny : module * ny + ny, :].astype(np.uint32), (pad + 1, pad)
        )
        _mask = np.pad(
            mask[module * ny : module * ny + ny, :].astype(np.uint32), (pad + 1, pad)
        )

        # compute interal images
        isum = kernel_summation(_image, knl)
        isum2 = kernel_summation(_image ** 2, knl)
        nsum = kernel_summation(_mask, knl)

        mean = isum / nsum
        variance = (isum2 - isum ** 2 / nsum) / nsum

        # TODO add the intensity of this pixel is over the threshold calc too
        signal[module * ny : module * ny + ny, :][
            (variance / mean) > 1 + sigma_s * np.sqrt(2 / (nsum - 1))
        ] = 1
    signal = signal * mask

    return signal


def simple():
    filename = sys.argv[1]
    setup(filename)

    m = mask()

    nz, ny, nx = shape()

    t0 = time.time()
    for j in range(nz):
        d = data(j)
        signal = thresholded_dispersion(d, m)
        print(f"{j} {np.count_nonzero(signal)}")
    t1 = time.time()
    print(f"Processing {nz} images took {(t1 - t0):.1f}s")



simple()
