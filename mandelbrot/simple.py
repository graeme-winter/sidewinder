import numpy as np

from __init__ import plot, field

IMAX = 0xFFFF


def mandelbrot(c):
    z = 0
    n = 0
    while abs(z) <= 2 and n < IMAX:
        z = z * z + c
        n += 1
    return n


def main():
    f = field(1024)
    nx, ny = f.shape
    m = np.empty(shape=f.shape, dtype=np.int32)
    for i in range(nx):
        for j in range(ny):
            m[i, j] = mandelbrot(f[i, j])
    m[m == IMAX] = 0
    plot(m, "simple.png")


if __name__ == "__main__":
    main()
