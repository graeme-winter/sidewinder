import math
import time

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_uniform_float32

import numpy as np

SEED = 42
THREADS = 64
BLOCKS = 24


@cuda.jit
def mc_pi(states, iterations, out):
    tid = cuda.grid(1)

    inside = 0
    for i in range(iterations):
        x = xoroshiro128p_uniform_float32(states, tid)
        y = xoroshiro128p_uniform_float32(states, tid)
        if x ** 2 + y ** 2 <= 1.0:
            inside += 1

    out[tid] = 4.0 * inside / iterations


states = create_xoroshiro128p_states(THREADS * BLOCKS, seed=SEED)
out = np.zeros(THREADS * BLOCKS, dtype=np.float64)

for ITER in 10000, 100000, 1000000, 10000000:
    mc_pi[BLOCKS, THREADS](states, ITER, out)
    pi = out.mean()
    sd_pi = math.sqrt(out.var() / len(out))
    print(f"{ITER} {math.log10(abs(pi - math.pi)):.2f} {math.log10(sd_pi):.2f}")
