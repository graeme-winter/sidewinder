import sys

import numpy as np
import pyopencl as cl

from __init__ import plot, field

IMAX = 0xFFFF


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
    if len(sys.argv) == 1:
        help()
        sys.exit(1)

    devices = get_devices()
    context = cl.Context(devices=[devices[int(sys.argv[1])]])
    queue = cl.CommandQueue(context)

    f = field(1024).astype(np.complex64)

    # input, output buffers
    _field = cl.Buffer(context, cl.mem_flags.READ_ONLY, f.size * f.itemsize)

    _image = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, f.size * np.dtype(np.int32).itemsize
    )

    # compile openCL program

    program = cl.Program(context, open("opencl_sp.cl", "r").read()).build()
    mandelbrot = program.mandelbrot

    mandelbrot.set_scalar_arg_dtypes([None, np.int32, np.int32, np.int32, None])

    # queue up copying input
    cl.enqueue_copy(queue, _field, f)

    # actual calculation
    mandelbrot(queue, f.shape, None, _field, f.shape[0], f.shape[1], IMAX, _image)

    # queue up copying output
    image = np.empty(shape=f.shape, dtype=np.int32)
    cl.enqueue_copy(queue, image, _image)

    image[image == IMAX] = 0

    plot(image, "opencl.png")


if __name__ == "__main__":
    main()
