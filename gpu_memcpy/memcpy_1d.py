# memcpy_1d.py
#
# 1D memcpy function on a GPU, to verify correct treatment of data in local
# memory

import sys
import time

import numpy as np
import pyopencl as cl

KERNEL = """
__kernel void memcpy_1d(const __global unsigned short *mem_in,
                        const int length,
                        __global unsigned short *mem_out) {

  __local unsigned short _mem[LOCAL_SIZE];
  
  int gid,  gsz;

  gid = get_global_id(0);
  gsz = get_global_size(0);

  int lid, lsz;

  lid = get_local_id(0);
  lsz = get_local_size(0);

  if (lid == 0) {
    for (int i = 0; i < lsz; i++) {
      if (gid + i < length) {
        _mem[i] = mem_in[gid + i];
      } else {
        _mem[i] = 0xffff;
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (gid < length) {
    mem_out[gid] = _mem[lid];
  }

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
    memcpy_1d = program.memcpy_1d

    memcpy_1d.set_scalar_arg_dtypes(
        [
            None,
            np.int32,
            None,
        ]
    )

    shape = 16 * 1024**2

    mem_in = np.random.randint(0, 256, size=shape, dtype=np.uint16)

    _mem_in = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, mem_in.size * np.dtype(mem_in.dtype).itemsize
    )
    _mem_out = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, mem_in.size * np.dtype(mem_in.dtype).itemsize
    )

    mem_out = np.zeros(shape=mem_in.shape, dtype=mem_in.dtype)

    cl.enqueue_copy(queue, _mem_in, mem_in)

    work = (shape,)
    group = (256,)
    evt = memcpy_1d(
        queue,
        work,
        group,
        _mem_in,
        shape,
        _mem_out,
    )
    evt.wait()

    cl.enqueue_copy(queue, mem_out, _mem_out)

    for j in range(shape):
        assert mem_in[j] == mem_out[j]


main()
