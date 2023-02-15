#include <unistd.h>

// fixed point integer arithmetic formulation
//
// use s7.24 formulation for bits => have enough
// bits to give ~ 7 decimal places of accuracy
// i.e. similar to floats

#define mul(a, b) (int)(((long long)a) * ((long long)b) >> 24)

unsigned short iter(int cr, int ci) {
  unsigned short count = 0;
  int zr = 0;
  int zi = 0;
  int tmp;

  while (count < 4096) {
    int zr2 = mul(zr, zr);
    int zi2 = mul(zi, zi);
    if ((zr2 + zi2) > (4 << 24)) {
      break;
    }
    count++;
    tmp = zr;
    zr = zr2 - zi2 + cr;
    zi = 2 * mul(tmp, zi) + ci;
  }
  return count;
}

int main() {
  for (int j = 0; j < 1280; j++) {
    for (int i = 0; i < 1280; i++) {
      int cr = -(2 << 24) + 0x8000 * i + 0x4000;
      int ci = -(5 << 22) + 0x8000 * j + 0x4000;
      unsigned short c = iter(cr, ci);
      write(1, (void *)&c, 2);
    }
  }
  return 0;
}
