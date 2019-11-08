from numpy import pi
from phi.physics.field import *


def diffuse(field, amount, substeps=1):
    assert isinstance(field, CenteredGrid)
    if field.extrapolation == 'periodic':
        frequencies = math.fft(math.to_complex(field.data))
        k = math.fftfreq(field.resolution, mode='square')
        fft_laplace = -(2 * pi) ** 2 * k
        diffuse_kernel = math.exp(fft_laplace * amount)
        data = math.ifft(frequencies * diffuse_kernel)
        data = math.real(data)
    else:
        data = field.data
        for i in range(substeps):
            data += amount / substeps * field.laplace()
    return field.with_data(data)
