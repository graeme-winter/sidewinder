__kernel void spot_finder(const __global unsigned short *image,
                          const __global unsigned char *mask, const int frames,
                          const int height, const int width, const int knl,
                          const float sigma_s, const float sigma_b,
                          __global unsigned char *signal) {

  __local unsigned short _image[LOCAL_SIZE];
  __local unsigned char _mask[LOCAL_SIZE];

  int gid[3], ggd[3];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);
  gid[2] = get_global_id(2);

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

  int knl2 = 2 * knl + 1;
  int nj = lsz[1] + knl2;
  int nk = lsz[2] + knl2;

  if (lid[0] == lid[1] == lid[2] == 0) {
    int off[3];
    off[0] = ggd[0] * lsz[0];
    off[1] = ggd[1] * lsz[1];
    off[2] = ggd[2] * lsz[2];
    for (int j = 0; j < nj; j++) {
      for (int k = 0; k < nk; k++) {
        int _j = j - knl;
        int _k = k - knl;
        if (((off[1] + _j) < 0) || ((off[2] + _k) < 0) ||
            ((off[1] + _j) >= height) || ((off[2] + _k) >= width)) {
          _image[j * nk + k] = 0;
          _mask[j * nk + k] = 0;
        } else {
          _image[j * nk + k] = image[(off[0] * height * width) +
                                     (off[1] + _j) * width + (off[2] + _k)];
          _mask[j * nk + k] = mask[(off[0] * height * width) +
                                   (off[1] + _j) * width + (off[2] + _k)];
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if ((gid[0] >= frames) || (gid[1] >= height) || (gid[2] >= width)) {
    return;
  }

  // local and global pixel locations for this worker
  int gpxl = gid[0] * width * height + gid[1] * width + gid[2];
  int lpxl = (lid[1] + knl) * nk + lid[2] + knl;

  // if masked, cannot be signal
  if (_mask[lpxl] == 0) {
    signal[gpxl] = 0;
    return;
  }

  float sum = 0.0;
  float sum2 = 0.0;
  float n = 0.0;

  for (int j = 0; j < knl2; j++) {
    for (int k = 0; k < knl2; k++) {
      int pxl = (lid[1] + j) * nk + lid[2] + k;
      sum += _image[pxl] * _mask[pxl];
      sum2 += _image[pxl] * _image[pxl] * _mask[pxl];
      n += _mask[pxl];
    }
  }

  if ((n >= 2) && (sum >= 0)) {
    float n_disp = n * sum2 - sum * sum - sum * (n - 1);
    float t_disp = sum * sigma_b * sqrt(2 * (n - 1));
    float n_stng = n * _image[lpxl] - sum;
    float t_stng = sigma_s * sqrt(n * sum);
    if ((n_disp > t_disp) && (n_stng > t_stng)) {
      signal[gpxl] = 1;
      return;
    }
  }

  signal[gpxl] = 0;

  return;
}
