import numpy as np

from numba import jit

from __init__ import plot, field

IMAX = 0xFFFF


@jit(nopython=True)
def mandelbrot(c):
    z = 0
    n = 0
    while abs(z) <= 2 and n < IMAX:
        z = z * z + c
        n += 1
    return n


def main():
    f = field(1024)
    v_mandelbrot = np.vectorize(mandelbrot)
    m = v_mandelbrot(f)
    m[m == IMAX] = 0
    plot(m, "numba_jit.png")


if __name__ == "__main__":
    main()
