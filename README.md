# sidewinder
Explorations on making Python fast, using tools such as Numba, Dask,
openCL etc.

Start off with a Mandelbrot set calculation as being (i) trivial (ii)
embarassingly parallel and (iii) easy to assess correctness.

Then look at some work with `pyopencl` for memcpu on GPU to ensure
that things move back and forth correctly.

Then look at some real work - using GPU for statistical feature
detection in X-ray diffraction images. 
