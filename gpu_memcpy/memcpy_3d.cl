__kernel void memcpy_3d(const __global unsigned short *mem_in, const int frames,
                        const int height, const int width,
                        __global unsigned short *mem_out) {

  __local unsigned short _mem[LOCAL_SIZE];

  int gid[3], gsz[3], ggd[3];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);
  gid[2] = get_global_id(2);

  gsz[0] = get_global_size(0);
  gsz[1] = get_global_size(1);
  gsz[2] = get_global_size(2);

  ggd[0] = get_group_id(0);
  ggd[1] = get_group_id(1);
  ggd[2] = get_group_id(2);

  int lid[3], lsz[3];

  lid[0] = get_local_id(0);
  lid[1] = get_local_id(1);
  lid[2] = get_local_id(2);

  lsz[0] = get_local_size(0);
  lsz[1] = get_local_size(1);
  lsz[2] = get_local_size(2);

  if (lid[0] == lid[1] == lid[2] == 0) {
    for (int i = 0; i < lsz[0]; i++) {
      for (int j = 0; j < lsz[1]; j++) {
        for (int k = 0; k < lsz[2]; k++) {
          if (((ggd[0] * lsz[0] + i) > frames) ||
              ((ggd[1] * lsz[1] + j) > height) ||
              ((ggd[2] * lsz[2] + k) > width)) {
            _mem[i * lsz[1] * lsz[2] + j * lsz[2] + k] = 0;
            continue;
          }
          _mem[i * lsz[1] * lsz[2] + j * lsz[2] + k] =
              mem_in[(ggd[0] * lsz[0] + i) * height * width +
                     (ggd[1] * lsz[1] + j) * width + (ggd[2] * lsz[2] + k)];
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if ((gid[0] < frames) && (gid[1] < height) && (gid[2] < width)) {
    mem_out[gid[0] * height * width + gid[1] * width + gid[2]] =
        _mem[lid[0] * lsz[1] * lsz[2] + lid[1] * lsz[2] + lid[2]];
  }

  return;
}
