import sys
import time
import math

import numpy as np
from matplotlib import pyplot as plt
import pyopencl as cl

from __init__ import setup, mask, data, shape, rettilb


def get_devices():
    result = []
    for pl in cl.get_platforms():
        result.extend(pl.get_devices())
    return result


def _help():
    devices = get_devices()
    print("Available devices:")
    for j, dev in enumerate(devices):
        print(f"{j} -> {dev.name}")
        print(f"Type:                 {cl.device_type.to_string(dev.type)}")
        print(f"Vendor:               {dev.vendor}")
        print(f"Available:            {bool(dev.available)}")
        print(f"Memory (global / MB): {int(dev.global_mem_size/1024**2)}")
        print(f"Memory (local / B):   {dev.local_mem_size}")
        print("")
    print(f"Please select device 0...{len(devices)-1}")


def main():
    if len(sys.argv) != 3:
        _help()
        sys.exit(1)

    filename = sys.argv[1]
    device = int(sys.argv[2])

    devices = get_devices()
    context = cl.Context(devices=[devices[device]])
    queue = cl.CommandQueue(context)

    max_group = devices[device].max_work_group_size
    max_item = devices[device].max_work_item_sizes

    program = cl.Program(context, open("index_dt.cl", "r").read()).build()
    index_dt = program.index_dt

    index_dt.set_scalar_arg_dtypes(
        [None, None, np.int32, np.int32, np.int32, np.float32, None]
    )

    setup(filename)

    m = mask()

    nz, ny, nx = shape()

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

    # copy input
    cl.enqueue_copy(queue, _image, d)
    cl.enqueue_copy(queue, _mask, m)

    # TODO consider 32 modules as 32 images

    # actual calculation - TODO consider waiting for event which is arrival
    # of data in memory (so make transfers above async)
    # re-arrange groups to have the longest axis _first_ - N.B. this has
    # nothing to do with memory (and is the reversed order) - also make the
    # "fast" direction slightly longer to allow nice divisors to make for a
    # boxy-ish workgroup
    index_dt(
        queue, (1040, 512, 32), (26, 8, 1), _image, _mask, 512, 1028, 7, 3.0, _signal
    )

    signal = np.empty(shape=m.shape, dtype=m.dtype)
    cl.enqueue_copy(queue, signal, _signal)

    signal_image = rettilb(signal)
    plt.imshow(signal_image, cmap="Greys")
    plt.show()


main()
