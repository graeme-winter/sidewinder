import sys
import time

from __init__ import setup, mask, data, shape


def simple():
    filename = sys.argv[1]
    setup(filename)

    m = mask()

    nz, ny, nx = shape()

    t0 = time.time()
    for j in range(nz):
        d = data(j)
    t1 = time.time()

    print(f"Time / frame (s): {(t1 - t0) / nz:.3f}")


simple()
