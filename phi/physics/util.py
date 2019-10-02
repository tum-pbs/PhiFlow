from phi import math
from numpy import pi


def diffuse(tensor, amount):
    frequencies = math.fft(math.to_complex(tensor))
    k = math.fftfreq(math.staticshape(tensor)[1:-1])
    fft_laplace = -(2 * pi) ** 2 * math.sum(k ** 2, axis=-1, keepdims=True)
    diffuse_kernel = math.exp(fft_laplace * amount)
    return math.ifft(frequencies * diffuse_kernel)