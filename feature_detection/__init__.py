import numpy as np
import h5py
from matplotlib import pyplot as plt

__hdf5 = None
__data = None
__mask = None

MOD_FAST = 1028
MOD_SLOW = 512
GAP_FAST = 12
GAP_SLOW = 38
N_FAST = 4
N_SLOW = 8


def rettilb(i_in):
    """Reverse the blitter operation, return an image from a stack of module
    images."""

    i_out = -1 * np.ones(
        shape=(
            N_SLOW * MOD_SLOW + (N_SLOW - 1) * GAP_SLOW,
            N_FAST * MOD_FAST + (N_FAST - 1) * GAP_FAST,
        ),
        dtype=i_in.dtype,
    )

    for n in range(N_SLOW * N_FAST):

        # sink
        s, f = divmod(n, N_FAST)
        s0 = s * (MOD_SLOW + GAP_SLOW)
        s1 = s0 + MOD_SLOW
        f0 = f * (MOD_FAST + GAP_FAST)
        f1 = f0 + MOD_FAST

        # source
        _s0 = n * MOD_SLOW
        _s1 = _s0 + MOD_SLOW
        _f0 = 0
        _f1 = _f0 + MOD_FAST
        i_out[s0:s1, f0:f1] = i_in[_s0:_s1, _f0:_f1]

    return i_out


def blitter(i_in):
    """Read the data from the 32 modules of the input image across to a
    big array with the 32 modules stacked on top of one another with 3 pixel
    border around every module."""

    assert i_in.shape == (
        N_SLOW * MOD_SLOW + (N_SLOW - 1) * GAP_SLOW,
        N_FAST * MOD_FAST + (N_FAST - 1) * GAP_FAST,
    )

    i_out = np.zeros(shape=(MOD_SLOW * (N_SLOW * N_FAST), MOD_FAST), dtype=i_in.dtype)

    for n in range(N_SLOW * N_FAST):

        # source
        s, f = divmod(n, N_FAST)
        s0 = s * (MOD_SLOW + GAP_SLOW)
        s1 = s0 + MOD_SLOW
        f0 = f * (MOD_FAST + GAP_FAST)
        f1 = f0 + MOD_FAST

        # sink
        _s0 = n * MOD_SLOW
        _s1 = _s0 + MOD_SLOW
        _f0 = 0
        _f1 = _f0 + MOD_FAST
        i_out[_s0:_s1, _f0:_f1] = i_in[s0:s1, f0:f1]

    return i_out


def setup(filename):
    """Set up reading the HDF5 file for input: assumes that you have virtual
    data sets configured ->

    /entry/data/data points at "all the data"
    /entry/instrument/detector/pixel_mask points at the mask

    ... and that is all we need."""

    global __hdf5, __data, __mask

    __hdf5 = h5py.File(filename, "r")
    __data = __hdf5["/entry/data/data"]

    nz, ny, nx = __data.shape

    tmp = __hdf5["/entry/instrument/detector/pixel_mask"][()]
    tmp = tmp.reshape((ny, nx))

    mask = np.zeros(tmp.shape, dtype=np.uint8)
    mask[tmp == 0] = 1

    __mask = blitter(mask)


def plot(signal):
    signal_image = rettilb(signal)
    plt.imshow(signal_image[:512, :1028], cmap="Greys")
    plt.show()


def data(frame):
    return blitter(__data[frame, :, :].astype(np.uint16))


def mask():
    return __mask


def shape():
    return __data.shape[0], __mask.shape[0], __mask.shape[1]
