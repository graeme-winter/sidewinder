#include <unistd.h>

unsigned short iter(float cr, float ci) {
  unsigned short count = 0;
  float zr = 0.0;
  float zi = 0.0;
  float tmp;

  while (count < 1000) {
    float zr2 = zr * zr;
    float zi2 = zi * zi;
    if ((zr2 + zi2) > 4) {
      break;
    }
    count++;
    tmp = zr;
    zr = zr2 - zi2 + cr;
    zi = 2 * tmp * zi + ci;
  }
  return count;
}

int main() {
  for (int j = 0; j < 1000; j++) {
    for (int i = 0; i < 1000; i++) {
      float cr = -2 + 0.0025 * i + 0.00125;
      float ci = -1.25 + 0.0025 * j + 0.00125;
      unsigned short c = iter(cr, ci);
      write(1, (void *)&c, 2);
    }
  }
  return 0;
}

