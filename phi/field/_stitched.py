from typing import Any

from phi import math
from phi.math import Tensor, channel, spatial, Shape, Extrapolation, batch
from phi.geom import GridCell, Box, Geometry

from ._field import Field
from ._grid import Grid, CenteredGrid


class StitchedGrid(Field):
    """
    Blocks laid out in a grid.
    Each block is a grid whose resolution can be changed at runtime.
    """

    def __init__(self,
                 values: Any,
                 extrapolation: Any = 0.,
                 bounds: Box = None,
                 resolution: Shape = None,
                 block_res: int or Shape = 32,
                 **resolution_: int or Tensor):
        """
        Args:
            values: Initial values, see `CenteredGrid` constructor.
            extrapolation: Extrapolation for the outer edges.
            bounds: Bounds of the total grid.
            resolution: Optional resolution as `Shape`, otherwise uses resolution keyword arguments.
            block_res: `int` or `Shape`
            **resolution_: Number of blocks along each spatial dimension.
        """
        resolution = (resolution or math.EMPTY_SHAPE) & spatial(**resolution_)
        bounds = bounds or Box(0, math.wrap(resolution, channel('vector')))
        self._blocks = GridCell(resolution, bounds)
        self._grids = {}
        for block_index in resolution.meshgrid():
            sub_bounds = self._blocks[block_index].bounds
            stitch = Stitch(self, block_index)
            sub_grid = CenteredGrid(values, stitch, sub_bounds, resolution=block_res)
            self._grids[spatial(**block_index)] = sub_grid
        self._resolution = resolution
        if not isinstance(extrapolation, math.Extrapolation):
            extrapolation = math.extrapolation.ConstantExtrapolation(extrapolation)
        self._extrapolation = extrapolation
        super().__init__(bounds)

    @property
    def shape(self) -> Shape:
        return self._resolution # & math.merge_shapes(grid.shape for grid in self.grids)

    @property
    def bounds(self) -> Box:
        return self._bounds

    @property
    def resolution(self) -> Shape:
        return self._resolution

    @property
    def extrapolation(self) -> Extrapolation:
        return self._extrapolation

    def _sample(self, geometry: Geometry) -> math.Tensor:
        raise NotImplementedError()

    def __getitem__(self, block_index: dict or Shape) -> 'Field':
        if not block_index:
            return self
        block_index = (block_index if isinstance(block_index, Shape) else spatial(**block_index))._reorder(self._resolution.names)
        return self._grids[block_index]

    @property
    def values(self):
        return math.stack([g.values for g in self._grids.values()], dim=batch('blocks'))

    def resample_block(self, block_index: dict or Shape, block_res: int or Shape):
        block_index = (block_index if isinstance(block_index, Shape) else spatial(**block_index))._reorder(self._resolution.names)
        old_grid = self[block_index]
        self._grids[block_index] = CenteredGrid(old_grid, old_grid.extrapolation, old_grid.bounds, resolution=block_res)


class Stitch(Extrapolation):

    def __init__(self, stitched_grid: StitchedGrid, block_index: dict):
        super().__init__(pad_rank=1)
        self.stitched_grid = stitched_grid
        self.block_index = block_index

    def to_dict(self) -> dict:
        return {'type': 'stitch'}

    def spatial_gradient(self) -> 'Extrapolation':
        raise NotImplementedError()

    def valid_outer_faces(self, dim) -> tuple:
        lower = self.stitched_grid.extrapolation.valid_outer_faces(dim)[0] if self.block_index[dim] == 0 else True
        upper = self.stitched_grid.extrapolation.valid_outer_faces(dim)[1] if self.block_index[dim] == self.stitched_grid.resolution.get_size(dim) - 1 else False
        return lower, upper

    @property
    def connects_to_outside(self) -> bool:
        return self.stitched_grid.resolution.volume > 1 or self.stitched_grid.extrapolation.connects_to_outside

    def pad_values(self, value: Tensor, width: int, dimension: str, upper_edge: bool) -> Tensor:
        self_grid = self.stitched_grid[self.block_index]
        block_index = {dim: (c + int(upper_edge) * 2 - 1 if dim == dimension else c) for dim, c in self.block_index.items()}
        if all([i >= 0 and i < self.stitched_grid.resolution.get_size(dim) for dim, i in block_index.items()]):
            neighbor_grid = self.stitched_grid[block_index]
            if self_grid.resolution == neighbor_grid.resolution:
                neighbor_slice = neighbor_grid.values[{dimension: slice(-1, None) if upper_edge else slice(0, 1)}]
                # values may need to be padded along other dimensions
                other_pad_widths = {dim.name: (1, 1) for dim in value.shape if dim.name != dimension and dim.size > neighbor_grid.resolution.get_size(dim)}
                if other_pad_widths:
                    neighbor_slice = math.extrapolation.BOUNDARY.pad(neighbor_slice, other_pad_widths)
                return neighbor_slice
            else:
                raise NotImplementedError()  # resample
        else:
            return self.stitched_grid.extrapolation.pad_values(value, width, dimension, upper_edge)

    def __repr__(self):
        return f"Stitch@{self.block_index}"
