import sys

import numpy as np

from scitbx.array_family import flex
import iotbx.phil
from dials.algorithms.spot_finding.factory import SpotFinderFactory
from dials.algorithms.spot_finding.factory import phil_scope as spot_phil

from spot_finder_data import setup, mask, data, shape, rettilb, plot

PHIL_SETTINGS = """
spotfinder {
  filter {
    min_spot_size = 1
  }
  threshold {
    algorithm = *dispersion
  }
}
"""


def find_signal_pixels(mask, image):
    """Mask: 2D array of unsigned char: if 0 -> invalid, if 1 -> valid,
    image: stack of 32 modules, so 3D numpy array of unsigned short. Returns
    3D array of unsigned char."""

    shape = mask.shape
    assert mask.shape == image.shape

    spot_params = spot_phil.fetch(
        source=iotbx.phil.parse("min_spot_size=1 algorithm=dispersion")
    ).extract()

    signal = np.empty(shape, dtype=np.uint8)
    threshold_function = SpotFinderFactory.configure_threshold(spot_params)

    for module in range(shape[0]):
        _mask = flex.int(mask[module, :, :].astype(np.int32)) == 1
        _image = flex.double(image[module, :, :].astype(np.float64))
        _signal = threshold_function.compute_threshold(_image, _mask)
        signal[module, :, :] = _signal.as_numpy_array().astype(np.uint8)

    return signal


def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    filename = sys.argv[1]
    setup(filename)

    m = mask().reshape(32, 512, 1028)

    nz, ny, nx = shape()

    for image in range(nz):
        d = data(image).reshape(32, 512, 1028)
        s = find_signal_pixels(m, d).reshape(16384, 1028)
        print(image, np.sum(s))


if __name__ == "__main__":
    main()
