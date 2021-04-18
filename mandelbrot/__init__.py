import numpy as np
from PIL import Image


def plot(image, filename):
    """Plot the image as a log2 greyscale image"""
    image = np.log2(1 + image.astype(np.float64))
    image *= 255.0 / np.max(image)
    Image.fromarray(image.astype(np.uint8)).save(filename)


def field(scl):
    """Generate a field of complex values with scale points / unit distance,
    from re = -2 to + 1, im over +/- 1.5. Thus total array size is
    9 * scl ^ 2"""

    x = 0.5 / scl
    real = np.outer(np.ones(3 * scl), np.arange(-2 + x, 1 + x, 1.0 / scl))
    imag = np.outer(np.arange(-1.5 + x, 1.5 + x, 1.0 / scl), np.ones(3 * scl))
    return real + 1j * imag
