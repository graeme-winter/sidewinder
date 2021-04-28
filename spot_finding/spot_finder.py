# spot_finder.py
#
# openCL / GPU powered spot finding
#

import sys
import time

import numpy as np
import pyopencl as cl

from spot_finder_data import setup, mask, data, shape, rettilb, plot
from spot_finder_cl import get_devices, device_help


def main():
    if len(sys.argv) != 3:
        device_help()
        sys.exit(1)

    filename = sys.argv[1]
    device = int(sys.argv[2])

    devices = get_devices()
    context = cl.Context(devices=[devices[device]])
    queue = cl.CommandQueue(context)

    # 16 box + 7 pixels around
    LOCAL = str((16 + 7) ** 2)

    cl_text = open("spot_finder.cl", "r").read().replace("LOCAL_SIZE", LOCAL)
    program = cl.Program(context, cl_text).build()
    spot_finder = program.spot_finder

    spot_finder.set_scalar_arg_dtypes(
        [
            None,
            None,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.float32,
            np.float32,
            None,
        ]
    )

    setup(filename)

    m = mask()

    nz, ny, nx = shape()

    data_shape = (32, ny // 32, nx)

    d = data(0)

    _image = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, d.size * np.dtype(d.dtype).itemsize
    )
    _mask = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, m.size * np.dtype(m.dtype).itemsize
    )
    _signal = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, m.size * np.dtype(m.dtype).itemsize
    )

    # mask same for all images -> only copy the once
    cl.enqueue_copy(queue, _mask, m)

    # likewise output buffer
    signal = np.zeros(shape=m.shape, dtype=m.dtype)

    group = (1, 16, 16)
    work = tuple(int(group[d] * np.ceil(data_shape[d] / group[d])) for d in (0, 1, 2))

    t0 = time.time()
    n = 0
    for i in range(nz):
        n += 1
        image = data(i)

        cl.enqueue_copy(queue, _image, image)
        evt = spot_finder(
            queue,
            work,
            group,
            _image,
            _mask,
            data_shape[0],
            data_shape[1],
            data_shape[2],
            3,
            3.0,
            6.0,
            _signal,
        )
        evt.wait()

        cl.enqueue_copy(queue, signal, _signal)
        print(i, np.count_nonzero(signal))

    t1 = time.time()
    print(f"Processing {n} images took {(t1 - t0):.1f}s")


main()
