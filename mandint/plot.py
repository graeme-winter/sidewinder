import numpy
from PIL import Image

ny = 1280
nx = 1280
x = numpy.fromfile("out", dtype=numpy.uint16, count=(nx * ny))
m = x.reshape((ny, nx))
m[m == 4096] = 0
m = numpy.sqrt(m)
image = (m * 4).astype(numpy.uint8)
Image.fromarray(image).save("mandel.png")
