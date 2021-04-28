import sys
import time
import math

import numpy as np
import pyopencl as cl
from matplotlib import pyplot as plt

KERNEL = """
__kernel void mtx(const __global unsigned short *image_in,
                  __global unsigned short *image_out) {

  __local unsigned short _image[LOCAL_SIZE];
  
  int gid[3],  gsz[3];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);
  gid[2] = get_global_id(2);

  gsz[0] = get_global_size(0);
  gsz[1] = get_global_size(1);
  gsz[2] = get_global_size(2);

  int lid[3], lsz[3];

  // lid[0] should be 0
  lid[0] = get_local_id(0);
  lid[1] = get_local_id(1);
  lid[2] = get_local_id(2);

  // lsz[0] should be 1
  lsz[0] = get_local_size(0);
  lsz[1] = get_local_size(1);
  lsz[2] = get_local_size(2);

  if (lid[0] == lid[1] == lid[2] == 0) {
    // it is my job to copy the data +/- knl size over from __global
    for (int i = 0; i < lsz[1]; i++) {
      int row = gid[1] + i;
      for (int j = 0; j < lsz[2]; j++) {
        int pxl = gid[2] + j;
        int m = gid[0];
        _image[i * lsz[2] + j] = image_in[m * gsz[1] * gsz[2] + row * gsz[2] + pxl];
      }
    }
  }

  // synchronise threads so all have same view of local memory
  barrier(CLK_LOCAL_MEM_FENCE);

  // local and global pixel locations for this worker
  int gpxl = gid[0] * gsz[1] * gsz[2] + gid[1] * gsz[2] + gid[2];
  int lpxl = lid[1] * lsz[2] + lid[2];

  image_out[gpxl] = _image[lpxl];

  return;
}
"""


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

    cl_text = KERNEL.replace("LOCAL_SIZE", "256")
    program = cl.Program(context, cl_text).build()
    mtx = program.mtx

    mtx.set_scalar_arg_dtypes(
        [
            None,
            None,
        ]
    )

    SHAPE = 2, 96, 128

    d = np.arange(SHAPE[0] * SHAPE[1] * SHAPE[2], dtype=np.int32) % 256
    d = d.reshape(SHAPE).astype(np.uint16)

    _image_in = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, d.size * np.dtype(d.dtype).itemsize
    )
    _image_out = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, d.size * np.dtype(d.dtype).itemsize
    )

    image_out = np.zeros(shape=d.shape, dtype=d.dtype)

    cl.enqueue_copy(queue, _image_in, d)

    work = SHAPE
    group = (1, 16, 16)
    evt = mtx(
        queue,
        work,
        group,
        _image_in,
        _image_out,
    )
    evt.wait()

    cl.enqueue_copy(queue, image_out, _image_out)

    plt.imshow(image_out[0, :, :], cmap="Greys")
    plt.show()


main()
