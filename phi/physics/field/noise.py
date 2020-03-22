import numpy as np
from phi import struct, math
from phi.physics.field import AnalyticField
from .grid import CenteredGrid
from .staggered_grid import StaggeredGrid
from ..domain import Domain


@struct.definition()
class Noise(AnalyticField):
    """
Generates random noise fluctuations which can be configured in physical size and smoothness.
Each call to at() or sample_at() generates a new noise field.
Noise can be used as an initializer for CenteredGrids or StaggeredGrids.
"""

    def __init__(self, **kwargs):
        AnalyticField.__init__(self, None, **struct.kwargs(locals()))

    def at(self, other_field):
        if isinstance(other_field, CenteredGrid):
            batch_size = other_field._batch_size
            if batch_size is None:
                batch_size = math.shape(other_field.data)[0]
            array = self.grid_sample(other_field.resolution, other_field.box.size, batch_size=batch_size, dtype=other_field.data.dtype)
            return other_field.with_data(array)
        if isinstance(other_field, StaggeredGrid):
            assert self.channels is None or self.channels == other_field.rank
            return other_field.with_data([self.at(grid) for grid in other_field.unstack()])
        if isinstance(other_field, Domain):
            array = self.grid_sample(other_field.resolution, other_field.box.size)
            return CenteredGrid(array, box=other_field.box, extrapolation='boundary')

    def sample_at(self, points):
        raise NotImplementedError()

    def grid_sample(self, resolution, size, batch_size=1, dtype=np.float32):
        shape = (batch_size,) + tuple(resolution) + (self.channels,)
        rndj = math.randn(shape, dtype) + 1j * math.randn(shape, dtype)
        k = math.fftfreq(resolution) * resolution / size  # in physical units
        k = math.sum(k ** 2, axis=-1, keepdims=True)
        k *= self.scale
        lowest_frequency = 0.1
        weight_mask = 1 / (1 + math.exp((lowest_frequency - k) * 1e3))  # High pass filter
        # --- Compute 1/k ---
        k[(0,) * len(k.shape)] = np.inf
        inv_k = 1 / k
        inv_k[(0,) * len(k.shape)] = 0
        # --- Compute result ---
        fft = rndj * inv_k ** self.smoothness * weight_mask
        array = math.real(math.ifft(fft)).astype(dtype)
        array /= math.std(array)
        array -= math.mean(array)
        return array

    @struct.constant()
    def channels(self, channels):
        """ Number of independent random scalar fields this Field consists of """
        return channels

    @struct.constant(default=100)
    def scale(self, scale):
        """ Size of noise fluctuations """
        return scale

    @struct.constant(default=1.0)
    def smoothness(self, smoothness):
        """ Determines how quickly high frequencies die out """
        return smoothness
