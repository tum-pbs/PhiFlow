import numpy as np

from phi import math
from phi.geom import GridCell
from phi.math import random_normal
from ._analytic import AnalyticField


class Noise(AnalyticField):
    """
Generates random noise fluctuations which can be configured in physical size and smoothness.
Each call to at() or sample_at() generates a new noise field.
Noise can be used as an initializer for CenteredGrids or StaggeredGrids.
"""

    def __init__(self, shape=None, scale=10, smoothness=1.0, batch=None, **channels):
        """

        :param channels: Number of independent random scalar fields this Field consists of
        :param scale: Size of noise fluctuations in physical units
        :param smoothness: Determines how quickly high frequencies die out
        """
        self.scale = scale
        self.smoothness = smoothness
        if shape is not None:
            self._shape = shape
            assert batch is None
            assert len(channels) == 0
        else:
            self._shape = math.batch_shape(batch) & math.channel_shape(channels)

    @property
    def shape(self):
        return self._shape

    def sample_at(self, points, reduce_channels=()):
        if isinstance(points, GridCell):
            return self.grid_sample(points.resolution, points.grid_size)
        raise NotImplementedError()

    def grid_sample(self, resolution: math.Shape, size, shape: math.Shape = None):
        shape = (self._shape if shape is None else shape).combined(resolution)
        rndj = math.to_complex(random_normal(shape)) + 1j * math.to_complex(random_normal(shape))  # Note: there is no complex32
        k = math.fftfreq(resolution) * resolution / size * self.scale  # in physical units
        k = math.vec_squared(k)
        lowest_frequency = 0.1
        weight_mask = 1 / (1 + math.exp((lowest_frequency - k) * 1e3))  # High pass filter
        # --- Compute 1/k ---
        k.tensor[(0,) * len(k.shape)] = np.inf
        inv_k = 1 / k
        inv_k.tensor[(0,) * len(k.shape)] = 0
        # --- Compute result ---
        fft = rndj * inv_k ** self.smoothness * weight_mask
        array = math.real(math.ifft(fft))
        array /= math.std(array, axis=array.shape.non_batch)
        array -= math.mean(array, axis=array.shape.non_batch)
        array = math.to_float(array)
        return array

    def unstack(self, dimension: str) -> tuple:
        count = self.shape.get_size(dimension)
        reduced_shape = self.shape.without(dimension)
        return (Noise(reduced_shape, self.scale, self.smoothness),) * count

    def __repr__(self):
        return "%s, scale=%f, smoothness=%f" % (self._shape, self.scale, self.smoothness)
