__kernel void memcpy_2d(const __global unsigned short *mem_in, const int height,
                        const int width, __global unsigned short *mem_out) {

  __local unsigned short _mem[LOCAL_SIZE];

  int gid[2], gsz[2], ggd[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);
  gsz[0] = get_global_size(0);
  gsz[1] = get_global_size(1);

  ggd[0] = get_group_id(0);
  ggd[1] = get_group_id(1);

  int lid[2], lsz[2];

  lid[0] = get_local_id(0);
  lid[1] = get_local_id(1);
  lsz[0] = get_local_size(0);
  lsz[1] = get_local_size(1);

  if (lid[0] == lid[1] == 0) {
    for (int i = 0; i < lsz[0]; i++) {
      for (int j = 0; j < lsz[1]; j++) {
        if (((ggd[0] * lsz[0] + i) > height) ||
            ((ggd[1] * lsz[1] + j) > width)) {
          _mem[i * lsz[1] + j] = 0;
          continue;
        }
        _mem[i * lsz[1] + j] =
            mem_in[(ggd[0] * lsz[0] + i) * width + (ggd[1] * lsz[1] + j)];
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if ((gid[0] < height) && (gid[1] < width)) {
    mem_out[gid[0] * width + gid[1]] = _mem[lid[0] * lsz[1] + lid[1]];
  }

  return;
}
