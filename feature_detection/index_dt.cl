__kernel void index_dt(const __global unsigned short *image,
                       const __global unsigned char *mask, const int height,
                       const int width, const int knl_width,
                       const float sigma_s, __global unsigned char *signal) {

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

  // 8+6 lines in local buffer: each line is 26 pixels + 6 padding
  __local unsigned short _image[L_BLOCK];
  __local unsigned char _mask[L_BLOCK];

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

  int sum = 0;
  int sum2 = 0;
  int n = 0;

  // now I am working with respect to lid i.e. +/- knl around lid in the
  // local address space - because the local memory is padded this is
  // guaranteed to be OK

  if (1 == 0) {
    int gpxl = gid[2] * width * height + gid[1] * width + gid[0];
    int pxl = (lid[1] + knl) * nj + lid[0] + knl;
    signal[gpxl] = _image[pxl] + 255;
    return;
  }

  for (int i = -knl; i < knl + 1; i++) {
    for (int j = -knl; j < knl + 1; j++) {
      int pxl = (lid[1] + i) * nj + lid[0] + j;
      int _i = _image[pxl];
      int _m = _mask[pxl];
      sum += _i * _m;
      sum2 += _i * _i * _m;
      n += _m;
    }
  }

  int gpxl = gid[2] * width * height + gid[1] * width + gid[0];

  if (n < 2) {
    signal[gpxl] = 0;
    return;
  }

  float mean = (float)sum / (float)n;
  float variance =
      ((float)sum2 - ((float)sum * (float)sum / (float)n)) / (float)n;

  if (mean <= 0) {
    signal[gpxl] = 0;
    return;
  }

  if ((variance / mean) > (1 + sigma_s * sqrt((float) (2.0 / (n - 1.0))))) {
    signal[gpxl] = 1;
  } else {
    signal[gpxl] = 0;
  }
}
