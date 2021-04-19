__kernel void mandelbrot(const __global float2 *field, const int width,
                         const int height, const int imax,
                         __global int *image) {
  int j = get_global_id(0);
  int i = get_global_id(1);

  if (j >= width || i >= height) {
    return;
  }

  float cr = field[j + i * width].s0;
  float ci = field[j + i * width].s1;

  float zr = 0.0;
  float zi = 0.0;

  float tr, ti;

  int n = 0;

  while (((zr * zr + zi * zi) <= 4) && (n < imax)) {
    tr = zr * zr - zi * zi + cr;
    ti = 2 * zr * zi + ci;
    zr = tr;
    zi = ti;
    n += 1;
  }

  image[j + i * width] = n;
}
