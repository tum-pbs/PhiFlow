from phi import math
from numpy import pi
from phi.field import *


def diffuse(field, amount):
    assert isinstance(field, CenteredGrid)
    frequencies = math.fft(math.to_complex(field.data))
    k = math.fftfreq(field.resolution)
    fft_laplace = -(2 * pi) ** 2 * math.sum(k ** 2, axis=-1, keepdims=True)
    diffuse_kernel = math.exp(fft_laplace * amount)
    return field.copied_with(data=math.ifft(frequencies * diffuse_kernel))