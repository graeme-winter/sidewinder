__kernel void index_dt(const __global unsigned short *image,
                       const __global unsigned short *mask,
                       const __local unsigned short *_image,
                       const __local unsigned short *_mask, const int n_modules,
                       const int height, const int width, const int kernel,
                       const float sigma_s,
                       const __global unsigned char *signal) {

  // pixel in global address space
  int i = get_global_id(0);
  int j = get_global_id(1);

  if (j >= width || i >= height) {
    return;
  }

  int knl = (kernel - 1) / 2;

  int sum, sum2, n;

  // pixel in local address space
  int k = get_local_id(0);
  int l = get_local_id(1);

  int nh = get_local_size(0);
  int nw = get_local_size(1);

  if (k == l == 0) {
    // it is my job to copy the data +/- kernel size over from __global
    for (int _i = 0; _i < nh + 2 * knl; i++) {
      int row = i + _i - knl;
      if (row < 0) {
        for (_j = 0; _j < width; _j++) {
          _image[_i * width + _j] = 0;
          _mask[_i * width + _j] = 0;
        }
      } else if (row >= height) {
        for (_j = 0; _j < width; _j++) {
          _image[_i * width + _j] = 0;
          _mask[_i * width + _j] = 0;
        }
      } else {
        for (_j = 0; _j < width; _j++) {
          _image[_i * width + _j] = image[row * width + j];
          _mask[_i * width + _j] = mask[row * width + j];
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int sum = 0;
  int sum2 = 0;
  int n = 0;

  for (int _i = -knl; _i < _knl + 1; _i++) {
    if ((_i < 0) || (_i >= nh)) {
      continue;
    }
    for (int _j = -knl; _j < _knl + 1; _j++) {
      if ((_j < 0) || (_j >= nw)) {
        continue;
      }
      int _s = _image[_i * width + j];
      int _m = _mask[_i * width + j];
      sum += _s * _m;
      sum2 += _s * _s * _m;
      n += _m;
    }
  }

  if (n < 2) {
    signal[i * width + j] = 0;
    return;
  }

  float mean = (float)sum / (float)n;
  float variance =
      ((float)sum2 - ((float)sum * (float)sum / float(n))) / (float)n;

  if (mean <= 0) {
    signal[i * width + j] = 0;
    return;
  }

  if ((variance / mean) > (1 + sigma_s * sqrt(2.0 / (n - 1.0)))) {
    signal[i * width + j] = 1;
  } else {
    signal[i * width + j] = 0;
  }
}
