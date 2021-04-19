import numpy as np

from __init__ import plot, field

IMAX = 0xFF


def mandelbrot(c):
    cr = c.real
    ci = c.imag
    zr = 0
    zi = 0
    n = 0
    while (zr * zr + zi * zi) <= 4 and n < IMAX:
        tr = zr * zr - zi * zi + cr
        ti = 2 * zr * zi + ci
        zr, zi = tr, ti
        n += 1
    return n


def main():
    f = field(128)
    nx, ny = f.shape
    m = np.empty(shape=f.shape, dtype=np.int32)
    for i in range(nx):
        for j in range(ny):
            m[i, j] = mandelbrot(f[i, j])
    m[m == IMAX] = 0
    plot(m, "real.png")


if __name__ == "__main__":
    main()
