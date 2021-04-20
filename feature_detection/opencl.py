import sys
import time
import math

import numpy as np
import pyopencl as cl

from __init__ import setup, mask, data, shape, rettilb


def get_devices():
    result = []
    for pl in cl.get_platforms():
        result.extend(pl.get_devices())
    return result


def help():
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
        help()
        sys.exit(1)

    filename = sys.argv[1]
    device = int(sys.argv[2])

    devices = get_devices()
    context = cl.Context(devices=[devices[device]])
    queue = cl.CommandQueue(context)

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

    program = cl.Program(context, open("index_dt.cl", "r").read()).build()
    index_dt = program.index_dt

    # copy input
    cl.enqueue_copy(queue, _image, d)
    cl.enqueue_copy(queue, _mask, m)

    # actual calculation
    index_dt(queue, d.shape, (4, d.shape[1]), _image, _mask, ny, nx, 7, 3.0, _signal)

    signal = np.empty(shape=mask.shape, dtype=mask.dtype)
    cl.enqueue_copy(queue, signal, _signal)

    signal_image = rettilb(signal)

    plt.imshow(signal_image, cmap="Greys")
    plt.show()


main()
