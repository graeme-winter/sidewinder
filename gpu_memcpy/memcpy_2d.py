# memcpy_2d.py
#
# 2D memcpy function on a GPU, to verify correct treatment of data in local
# memory

import sys
import time

import numpy as np
import pyopencl as cl


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
    if len(sys.argv) != 2:
        _help()
        sys.exit(1)

    device = int(sys.argv[1])

    devices = get_devices()
    context = cl.Context(devices=[devices[device]])
    queue = cl.CommandQueue(context)

    # TODO use these to decide what "shape" to make the work groups
    # and add something which allows those shapes to be replaced in the
    # openCL source code
    max_group = devices[device].max_work_group_size
    max_item = devices[device].max_work_item_sizes

    cl_text = open("memcpy_2d.cl", "r").read().replace("LOCAL_SIZE", "256")
    program = cl.Program(context, cl_text).build()
    memcpy_2d = program.memcpy_2d

    memcpy_2d.set_scalar_arg_dtypes(
        [
            None,
            np.int32,
            np.int32,
            None,
        ]
    )

    shape = (3070, 4090)
    size = shape[0] * shape[1]

    mem_in = np.random.randint(0, 256, size=size, dtype=np.uint16).reshape(shape)

    _mem_in = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, mem_in.size * np.dtype(mem_in.dtype).itemsize
    )
    _mem_out = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, mem_in.size * np.dtype(mem_in.dtype).itemsize
    )

    mem_out = np.zeros(shape=mem_in.shape, dtype=mem_in.dtype)

    cl.enqueue_copy(queue, _mem_in, mem_in)

    # work must be a multiple of group size
    group = (12, 16)
    work = tuple(int(group[d] * np.ceil(shape[d] / group[d])) for d in (0, 1))
    print(f"{shape} -> {work}")
    evt = memcpy_2d(
        queue,
        work,
        group,
        _mem_in,
        shape[0],
        shape[1],
        _mem_out,
    )
    evt.wait()

    cl.enqueue_copy(queue, mem_out, _mem_out)

    assert np.array_equal(mem_in, mem_out)


main()
