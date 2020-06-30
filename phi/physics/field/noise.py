import numpy as np
from phi import struct, math
from phi.physics.field import AnalyticField
from .grid import CenteredGrid
from .staggered_grid import StaggeredGrid
from ..domain import Domain
from ...backend.backend import Backend


@struct.definition()
class Noise(AnalyticField):
    """
Generates random noise fluctuations which can be configured in physical size and smoothness.
Each call to at() or sample_at() generates a new noise field.
Noise can be used as an initializer for CenteredGrids or StaggeredGrids.
"""

    def __init__(self, channels=1, scale=10, smoothness=1.0, math=math.DYNAMIC_BACKEND, **kwargs):
        AnalyticField.__init__(self, None, **struct.kwargs(locals()))

    @struct.constant()
    def channels(self, channels):
        """ Number of independent random scalar fields this Field consists of """
        return channels

    @struct.constant()
    def scale(self, scale):
        """ Size of noise fluctuations """
        return scale

    @struct.constant()
    def smoothness(self, smoothness):
        """ Determines how quickly high frequencies die out """
        return smoothness

    def at(self, other_field):
        if isinstance(other_field, CenteredGrid):
            batch_size = other_field._batch_size
            if batch_size is None:
                if other_field.content_type in (struct.shape, struct.staticshape):
                    batch_size = other_field.data[0]
                else:
                    batch_size = math.shape(other_field.data)[0]
            array = self.grid_sample(other_field.resolution, other_field.box.size, batch_size=batch_size)
            return other_field.with_data(array)
        if isinstance(other_field, StaggeredGrid):
            assert self.channels is None or self.channels == other_field.rank
            return other_field.with_data([self.grid_sample(grid.resolution, grid.box.size, grid._batch_size, 1) for grid in other_field.unstack()])
        if isinstance(other_field, Domain):
            array = self.grid_sample(other_field.resolution, other_field.box.size)
            return CenteredGrid(array, box=other_field.box, extrapolation='boundary')

    def sample_at(self, points):
        raise NotImplementedError()

    def grid_sample(self, resolution, size, batch_size=1, channels=None):
        channels = channels or self.channels or len(size)
        shape = (batch_size,) + tuple(resolution) + (channels,)
        rndj = math.to_complex(self.math.random_normal(shape)) + 1j * math.to_complex(self.math.random_normal(shape))  # Note: there is no complex32
        k = math.fftfreq(resolution) * resolution / size * self.scale  # in physical units
        k = math.sum(k ** 2, axis=-1, keepdims=True)
        lowest_frequency = 0.1
        weight_mask = 1 / (1 + math.exp((lowest_frequency - k) * 1e3))  # High pass filter
        # --- Compute 1/k ---
        k[(0,) * len(k.shape)] = np.inf
        inv_k = 1 / k
        inv_k[(0,) * len(k.shape)] = 0
        # --- Compute result ---
        fft = rndj * inv_k ** self.smoothness * weight_mask
        array = math.real(math.ifft(fft))
        array /= math.std(array, axis=tuple(range(1, math.ndims(array))), keepdims=True)
        array -= math.mean(array, axis=tuple(range(1, math.ndims(array))), keepdims=True)
        array = math.to_float(array)
        return array

    @property
    def component_count(self):
        return self.channels

    @struct.constant()
    def math(self, m):
        assert isinstance(m, math.Backend)
        return m
