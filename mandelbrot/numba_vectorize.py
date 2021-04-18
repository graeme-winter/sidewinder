from numba import vectorize, int32, complex128

from __init__ import plot, field

IMAX = 0xFFFF


@vectorize([int32(complex128)], target="parallel")
def mandelbrot_vector(c):
    z = 0
    n = 0
    while abs(z) <= 2 and n < IMAX:
        z = z * z + c
        n += 1
    return n


def main():
    f = field(1024)
    m = mandelbrot_vector(f)
    m[m == IMAX] = 0
    plot(m, "numba_vectorize.png")


if __name__ == "__main__":
    main()
