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
from spot_finder_config import get_config


def main():
    if len(sys.argv) < 3:
        print(f"{sys.argv[0]} /path/to/data.nxs out.dat")
        sys.exit(1)

    filename = sys.argv[1]

    config = get_config()
    gpus = tuple(map(int, config["devices"].split(",")))

    devices = get_devices()
    context = cl.Context(devices=[devices[gpus[0]]])
    queue = cl.CommandQueue(context)

    # TODO verify that there is enough local memory for this size of work group
    # TODO verify that this size of work group is legal for this device
    local_work = tuple(map(int, config["work"].split(",")))

    # work box + 7 pixels around
    LOCAL = str((local_work[0] + 7) * (local_work[1] + 7))

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

    _mask = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, m.size * np.dtype(m.dtype).itemsize
    )

    # mask same for all images -> only copy the once
    cl.enqueue_copy(queue, _mask, m)

    _image0 = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, d.size * np.dtype(d.dtype).itemsize
    )
    _signal0 = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, m.size * np.dtype(m.dtype).itemsize
    )

    _image1 = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, d.size * np.dtype(d.dtype).itemsize
    )
    _signal1 = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, m.size * np.dtype(m.dtype).itemsize
    )

    # likewise output buffer
    signal0 = np.zeros(shape=m.shape, dtype=m.dtype)
    signal1 = np.zeros(shape=m.shape, dtype=m.dtype)

    bad = np.zeros(shape=d.shape, dtype=d.dtype)

    group = (1, local_work[0], local_work[1])
    work = tuple(int(group[d] * np.ceil(data_shape[d] / group[d])) for d in (0, 1, 2))

    t0 = time.time()
    n = 0
    for i in range(0, nz, 2):
        n += 1
        cl.enqueue_copy(queue, _image0, data(i))
        cl.enqueue_copy(queue, _image1, data(i + 1))
        evt0 = spot_finder(
            queue,
            work,
            group,
            _image0,
            _mask,
            data_shape[0],
            data_shape[1],
            data_shape[2],
            3,
            3.0,
            6.0,
            _signal0,
        )
        evt1 = spot_finder(
            queue,
            work,
            group,
            _image1,
            _mask,
            data_shape[0],
            data_shape[1],
            data_shape[2],
            3,
            3.0,
            6.0,
            _signal1,
        )
        evt0.wait()
        evt1.wait()

        cl.enqueue_copy(queue, signal0, _signal0)
        cl.enqueue_copy(queue, signal1, _signal1)
        bad += signal0
        bad += signal1

    t1 = time.time()

    print(f"Processing {n} images took {(t1 - t0):.1f}s")
    print(f"Found {np.count_nonzero(bad > n // 4)} bad pixels")

    with open(sys.argv[2], "w") as fout:
        bad = rettilb(bad > n // 4)
        bad[bad == -1] = 0
        bad = np.nonzero(bad)

        bad_slow = bad[0]
        bad_fast = bad[1]

        for s, f in zip(bad_slow, bad_fast):
            fout.write(f"{s} {f}\n")


main()
