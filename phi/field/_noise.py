import warnings

from phi import math
from phi.geom import UniformGrid, Geometry
from phi.math import random_normal, Tensor, channel, dual, Extrapolation, vec
from ._field import FieldInitializer


class Noise(FieldInitializer):
    """
    Generates random noise fluctuations which can be configured in physical size and smoothness.
    Each time values are sampled from a Noise field, a new noise field is generated.

    Noise is typically used as an initializer for CenteredGrids or StaggeredGrids.
    """

    def __init__(self, *shape: math.Shape, scale=10., smoothness=1.0, **channel_dims):
        """
        Args:
          shape: Batch and channel dimensions. Spatial dimensions will be added automatically once sampled on a grid.
          scale: Size of noise fluctuations in physical units.
          smoothness: Determines how quickly high frequencies die out.
          **dims: Additional dimensions, added to `shape`.
        """
        self.scale = scale
        self.smoothness = smoothness
        self._shape = math.concat_shapes(*shape, channel(**channel_dims))

    def _sample(self, geometry: Geometry, at: str, boundaries: Extrapolation, **kwargs) -> Tensor:
        if isinstance(geometry, UniformGrid):
            if at == 'center':
                return self.grid_sample(geometry.resolution, geometry.grid_size)
            elif at == 'face':
                result = {dim: self.grid_sample(grid.resolution, grid.grid_size) for dim, grid in geometry.staggered_cells(boundaries).items()}
                return vec(geometry.face_shape.dual, **result)
        raise NotImplementedError(f"{type(geometry)} not supported. Only UniformGrid allowed.")

    def grid_sample(self, resolution: math.Shape, size, shape: math.Shape = None):
        shape = (self._shape if shape is None else shape) & resolution
        for dim in channel(self._shape):
            if dim.name == 'vector' and dim.item_names[0] is None:
                warnings.warn(f"Please provide item names for Noise dim {dim} using {dim}='x,y,z'", FutureWarning)
                shape &= channel(**{dim.name: resolution.names})
        rndj = math.to_complex(random_normal(shape)) + 1j * math.to_complex(random_normal(shape))  # Note: there is no complex32
        # --- Compute 1 / k^2 ---
        k_vec = math.fftfreq(resolution, size) * resolution * math.tensor(self.scale)  # in physical units
        k2 = math.vec_squared(k_vec)
        lowest_frequency = 0.1
        weight_mask = math.to_float(k2 > lowest_frequency)
        inv_k2 = math.divide_no_nan(1, k2)
        # --- Compute result ---
        fft = rndj * inv_k2 ** self.smoothness * weight_mask
        array = math.real(math.ifft(fft))
        array /= math.std(array, dim=array.shape.non_batch)
        array -= math.mean(array, dim=array.shape.non_batch)
        array = math.to_float(array)
        return array

    def __repr__(self):
        return f"{self._shape}, scale={self.scale}, smoothness={self.smoothness}"
