from numpy import pi
from phi import math
from phi.geom import AABox
from .field import StaggeredSamplePoints
from .grid import CenteredGrid


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


def data_bounds(field):
    assert field.has_points
    try:
        data = field.points.data
        min_vec = math.min(data, axis=tuple(range(len(data.shape)-1)))
        max_vec = math.max(data, axis=tuple(range(len(data.shape)-1)))
    except StaggeredSamplePoints:
        boxes = [data_bounds(c) for c in field.unstack()]
        min_vec = math.min([b.lower for b in boxes], axis=0)
        max_vec = math.max([b.upper for b in boxes], axis=0)
    return AABox(min_vec, max_vec)