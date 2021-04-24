__kernel void index_dt(const __global unsigned short *image,
                       const __global unsigned char *mask,
                       __local unsigned short *_image,
                       __local unsigned char *_mask, const int height,
                       const int width, const int knl_width,
                       const float sigma_s, const float sigma_b,
		       __global unsigned char *signal) {

  // pixel in global address space - N.B. because reasons the work
  // assignment is _not_ ordered in the same way as the memory.
  int gid[3];
  gid[0] = get_global_id(0); // pixel number
  gid[1] = get_global_id(1); // row number
  gid[2] = get_global_id(2); // module number

  // hard coded for 32 modules at the moment FIXME get the true shape from
  // the API - we do ask for work units off the edge though so this is legit
  if (gid[2] >= 32 || gid[1] >= height || gid[0] >= width) {
    return;
  }

  int lid[3], lsz[3];
  lid[0] = get_local_id(0);
  lid[1] = get_local_id(1);
  lid[2] = get_local_id(2);

  // lsz[2] should be 1
  lsz[0] = get_local_size(0);
  lsz[1] = get_local_size(1);
  lsz[2] = get_local_size(2);

  int knl = (knl_width - 1) / 2;

  // loop variables - will always be consistent here that we have i, j, m where
  // m is the module number, i is the row and j is the pixel number - in this
  // address space I am working in global coordinates for the initial offset
  // and then local coordinates for the actual calculation - nota bene.

  int ni = lsz[1] + 2 * knl;
  int nj = lsz[0] + 2 * knl;

  if (lid[0] == lid[1] == lid[2] == 0) {
    // it is my job to copy the data +/- knl_width size over from __global
    for (int i = 0; i < ni; i++) {
      int row = gid[1] + i - knl;
      if (row < 0 || row >= height) {
        for (int j = 0; j < nj; j++) {
          _image[i * nj + j] = 0;
          _mask[i * nj + j] = 0;
        }
        continue;
      }
      for (int j = 0; j < nj; j++) {
        int pxl = gid[0] + j - knl;
        if (pxl < 0 || pxl >= width) {
          _image[i * nj + j] = 0;
          _mask[i * nj + j] = 0;
        } else {
          int m = gid[2];
          _image[i * nj + j] = image[m * width * height + row * width + pxl];
          _mask[i * nj + j] = mask[m * width * height + row * width + pxl];
        }
      }
    }
  }

  // synchronise threads so all have same view of local memory
  barrier(CLK_LOCAL_MEM_FENCE);

  // local and global pixel locations for this worker
  int gpxl = gid[2] * width * height + gid[1] * width + gid[0];
  int lpxl = (lid[1] + knl) * nj + lid[0] + knl;

  // if masked, cannot be signal
  if (_mask[lpxl] == 0) {
    signal[gpxl] = 0;
    return;
  }

  float sum = 0.0;
  float sum2 = 0.0;
  float n = 0.0;

  // now I am working with respect to lid i.e. +/- knl around lid in the
  // local address space - because the local memory is padded this is
  // guaranteed to be OK

  for (int i = 0; i < 2 * knl + 1; i++) {
    for (int j = 0; j < 2 * knl + 1; j++) {
      int pxl = (lid[1] + i) * nj + lid[0] + j;
      int _i = _image[pxl];
      int _m = _mask[pxl];
      sum += _i * _m;
      sum2 += _i * _i * _m;
      n += _m;
    }
  }

  if ((n >= 2) && (sum > 0)) {
    float n_disp = n * sum2 - sum * sum - sum * (n - 1);
    float t_disp = sum * sigma_b * sqrt(2 / (n - 1));
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
