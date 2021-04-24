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

    # TODO use these to decide what "shape" to make the work groups
    # and add something which allows those shapes to be replaced in the
    # openCL source code
    max_group = devices[device].max_work_group_size
    max_item = devices[device].max_work_item_sizes

    cl_text = open("index_dt.cl", "r").read()
    program = cl.Program(context, cl_text).build()
    index_dt = program.index_dt

    index_dt.set_scalar_arg_dtypes(
        [
            None,
            None,
            None,
            None,
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

    d = data(0)

    t0 = time.time()
    _image = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, d.size * np.dtype(d.dtype).itemsize
    )
    _mask = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, m.size * np.dtype(m.dtype).itemsize
    )
    _signal = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, m.size * np.dtype(m.dtype).itemsize
    )

    # mask, local memory struct same for all images -> only copy the once
    cl.enqueue_copy(queue, _mask, m)
    _limage = cl.LocalMemory(448 * 2)
    _lmask = cl.LocalMemory(448)

    # likewise output buffer
    signal = np.empty(shape=m.shape, dtype=m.dtype)

    # spot find on all the images...
    for image in range(nz):
        d = data(image)

        # copy input
        cl.enqueue_copy(queue, _image, d)

        # actual calculation - TODO consider waiting for event which is
        # arrival of data in memory (so make transfers above async)
        # re-arrange groups to have the longest axis _first_ - N.B. this has
        # nothing to do with memory (and is the reversed order)
        # "fast" direction slightly longer to allow nice divisors to make
        # for a boxy-ish workgroup
        work = (1040, 512, 32)
        group = (26, 8, 1)
        evt = index_dt(
            queue,
            work,
            group,
            _image,
            _mask,
            _limage,
            _lmask,
            512,
            1028,
            7,
            3.0,
            6.0,
            _signal,
        )
        evt.wait()

        cl.enqueue_copy(queue, signal, _signal)

        print(image, np.count_nonzero(signal))

    t1 = time.time()

    print(f"{nz} images took {(t1 - t0):.2f}s -> {nz / (t1 - t0):.1f}/s")



main()
