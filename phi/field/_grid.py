from typing import Any, Union

from phi import math
from phi.geom import Box, Geometry, UniformGrid
from phi.math import rename_dims
from ._field import Field, as_boundary, FieldInitializer
from ._resample import sample, sample_function
from ..math import Shape
from phiml.math._shape import spatial, channel, dual
from phiml.math._tensors import TensorStack, Tensor
from ..math.extrapolation import Extrapolation


def grid(values: Any = 0., extrapolation: Any = 0., bounds: Box or float = None, resolution: int or Shape = None, staggered=False, **resolution_: int or Tensor) -> Field:
    if staggered:
        return StaggeredGrid(values, extrapolation, bounds, resolution, **resolution_)
    else:
        return CenteredGrid(values, extrapolation, bounds, resolution, **resolution_)


def CenteredGrid(values: Any = 0.,
                 boundary: Any = 0.,
                 bounds: Box or float = None,
                 resolution: int or Shape = None,
                 extrapolation: Any = None,
                 convert=True,
                 **resolution_: int or Tensor) -> Field:
    """
    Create an n-dimensional grid with values sampled at the cell centers.
    A centered grid is defined through its `CenteredGrid.values` `phi.math.Tensor`, its `CenteredGrid.bounds` `phi.geom.Box` describing the physical size, and its `CenteredGrid.extrapolation` (`phi.math.extrapolation.Extrapolation`).
    
    Centered grids support batch, spatial and channel dimensions.

    See Also:
        `StaggeredGrid`,
        `Grid`,
        `Field`,
        `Field`,
        module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html

    Args:
        values: Values to use for the grid.
            Has to be one of the following:

            * `phi.geom.Geometry`: sets inside values to 1, outside to 0
            * `Field`: resamples the Field to the staggered sample points
            * `Number`: uses the value for all sample points
            * `tuple` or `list`: interprets the sequence as vector, used for all sample points
            * `phi.math.Tensor` compatible with grid dims: uses tensor values as grid values
            * Function `values(x)` where `x` is a `phi.math.Tensor` representing the physical location.
                The spatial dimensions of the grid will be passed as batch dimensions to the function.

        extrapolation: The grid extrapolation determines the value outside the `values` tensor.
            Allowed types: `float`, `phi.math.Tensor`, `phi.math.extrapolation.Extrapolation`.
        bounds: Physical size and location of the grid as `phi.geom.Box`.
            If the resolution is determined through `resolution` of `values`, a `float` can be passed for `bounds` to create a unit box.
        resolution: Grid resolution as purely spatial `phi.math.Shape`.
            If `bounds` is given as a `Box`, the resolution may be specified as an `int` to be equal along all axes.
        **resolution_: Spatial dimensions as keyword arguments. Typically either `resolution` or `spatial_dims` are specified.
        convert: Whether to convert `values` to the default backend.
    """
    extrapolation = boundary if extrapolation is None else extrapolation
    if resolution is None and not resolution_:
        assert isinstance(values, math.Tensor), "Grid resolution must be specified when 'values' is not a Tensor."
        resolution = values.shape.spatial
        elements = UniformGrid(resolution, bounds)
    else:
        resolution = _get_resolution(resolution, resolution_, bounds)
        elements = UniformGrid(resolution, bounds)
        if isinstance(values, math.Tensor):
            values = math.expand(values, resolution)
        elif isinstance(values, (Field, FieldInitializer, Geometry)):
            values = sample(values, elements)
        elif callable(values):
            values = sample_function(values, elements, 'center', extrapolation)
        else:
            if isinstance(values, (tuple, list)) and len(values) == resolution.rank:
                values = math.tensor(values, channel(vector=resolution.names))
            values = math.expand(math.tensor(values, convert=convert), resolution)
    if values.dtype.kind not in (float, complex):
        values = math.to_float(values)
    assert resolution.spatial_rank == elements.bounds.spatial_rank, f"Resolution {resolution} does not match bounds {bounds}"
    assert values.shape.spatial_rank == elements.spatial_rank, f"Spatial dimensions of values ({values.shape}) do not match elements {elements}"
    assert values.shape.spatial_rank == elements.bounds.spatial_rank, f"Spatial dimensions of values ({values.shape}) do not match elements {elements}"
    assert values.shape.instance_rank == 0, f"Instance dimensions not supported for grids. Got values with shape {values.shape}"
    return Field(elements, values, extrapolation)


def StaggeredGrid(values: Any = 0.,
                  boundary: float or Extrapolation = 0,
                  bounds: Box or float = None,
                  resolution: Shape or int = None,
                  extrapolation: float or Extrapolation = None,
                  convert=True,
                  **resolution_: int or Tensor) -> Field:
    """
    N-dimensional grid whose vector components are sampled at the respective face centers.
    A staggered grid is defined through its values tensor, its bounds describing the physical size, and its extrapolation.

    Staggered grids support batch and spatial dimensions but only one channel dimension for the staggered vector components.

    See Also:
        `CenteredGrid`,
        `Grid`,
        `Field`,
        `Field`,
        module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html

    Args:
        values: Values to use for the grid.
            Has to be one of the following:

            * `phi.geom.Geometry`: sets inside values to 1, outside to 0
            * `Field`: resamples the Field to the staggered sample points
            * `Number`: uses the value for all sample points
            * `tuple` or `list`: interprets the sequence as vector, used for all sample points
            * `phi.math.Tensor` with staggered shape: uses tensor values as grid values.
              Must contain a `vector` dimension with each slice consisting of one more element along the dimension they describe.
              Use `phi.math.stack()` to manually create this non-uniform tensor.
            * Function `values(x)` where `x` is a `phi.math.Tensor` representing the physical location.
                The spatial dimensions of the grid will be passed as batch dimensions to the function.

        boundary: The grid extrapolation determines the value outside the `values` tensor.
            Allowed types: `float`, `phi.math.Tensor`, `phi.math.extrapolation.Extrapolation`.
        bounds: Physical size and location of the grid as `phi.geom.Box`.
            If the resolution is determined through `resolution` of `values`, a `float` can be passed for `bounds` to create a unit box.
        resolution: Grid resolution as purely spatial `phi.math.Shape`.
            If `bounds` is given as a `Box`, the resolution may be specified as an `int` to be equal along all axes.
        convert: Whether to convert `values` to the default backend.
        **resolution_: Spatial dimensions as keyword arguments. Typically either `resolution` or `spatial_dims` are specified.
    """
    extrapolation = boundary if extrapolation is None else extrapolation
    extrapolation = as_boundary(extrapolation, UniformGrid)
    if resolution is None and not resolution_:
        assert isinstance(values, Tensor), "Grid resolution must be specified when 'values' is not a Tensor."
        if not all(extrapolation.valid_outer_faces(d)[0] != extrapolation.valid_outer_faces(d)[1] for d in spatial(values).names):  # non-uniform values required
            if '~vector' not in values.shape:
                values = unstack_staggered_tensor(values, extrapolation)
            resolution = resolution_from_staggered_tensor(values, extrapolation)
        else:
            resolution = spatial(values)
        bounds = bounds or Box(math.const_vec(0, resolution), math.wrap(resolution, channel('vector')))
        elements = UniformGrid(resolution, bounds)
    else:
        resolution = _get_resolution(resolution, resolution_, bounds)
        elements = UniformGrid(resolution, bounds)
        if isinstance(values, math.Tensor):
            if not spatial(values):
                values = expand_staggered(values, resolution, extrapolation)
            if not all(extrapolation.valid_outer_faces(d)[0] != extrapolation.valid_outer_faces(d)[1] for d in resolution.names):  # non-uniform values required
                if '~vector' not in values.shape:  # legacy behavior: we are given a padded staggered tensor
                    values = unstack_staggered_tensor(values, extrapolation)
                    resolution = resolution_from_staggered_tensor(values, extrapolation)
                    elements = UniformGrid(resolution, bounds)
                else:  # Keep dim order from data and check it matches resolution
                    assert set(resolution_from_staggered_tensor(values, extrapolation)) == set(resolution), f"Failed to create StaggeredGrid: values {values.shape} do not match given resolution {resolution} for extrapolation {extrapolation}. See https://tum-pbs.github.io/PhiFlow/Staggered_Grids.html"
        elif isinstance(values, (Geometry, Field, FieldInitializer)):
            values = sample(values, elements, at='face', boundary=extrapolation, dot_face_normal=elements)
        elif callable(values):
            values = sample_function(values, elements, 'face', extrapolation)
            if elements.shape.is_non_uniform:  # Different number of X and Y faces
                assert isinstance(values, TensorStack), f"values function must return a staggered Tensor but returned {type(values)}"
            assert '~vector' in values.shape
            if 'vector' in values.shape:
                values = math.stack([values[{'vector': i, '~vector': i}] for i in range(resolution.rank)], dual(vector=resolution))
        else:
            values = expand_staggered(math.tensor(values, convert=convert), resolution, extrapolation)
    if values.dtype.kind not in (float, complex):
        values = math.to_float(values)
    assert resolution.spatial_rank == elements.bounds.spatial_rank, f"Resolution {resolution} does not match bounds {elements.bounds}"
    if 'vector' in values.shape:
        values = rename_dims(values, 'vector', dual(vector=values.vector.item_names))
    assert values.shape.spatial_rank == elements.spatial_rank, f"Spatial dimensions of values ({values.shape}) do not match elements {elements}"
    assert values.shape.spatial_rank == elements.bounds.spatial_rank, f"Spatial dimensions of values ({values.shape}) do not match elements {elements}"
    assert values.shape.instance_rank == 0, f"Instance dimensions not supported for grids. Got values with shape {values.shape}"
    return Field(elements, values, extrapolation)


def unstack_staggered_tensor(data: Tensor, extrapolation: Extrapolation) -> TensorStack:
    sliced = []
    assert 'vector' in data.shape
    for dim, component in zip(data.shape.spatial.names, data.vector):
        lo_valid, up_valid = extrapolation.valid_outer_faces(dim)
        slices = {d: slice(0, -1) for d in data.shape.spatial.names}
        slices[dim] = slice(int(not lo_valid), - int(not up_valid) or None)
        sliced.append(component[slices])
    return math.stack(sliced, dual(vector=spatial(data)))


def expand_staggered(values: Tensor, resolution: Shape, extrapolation: Extrapolation):
    """ Add missing spatial dimensions to `values` """
    cells = UniformGrid(resolution, Box(math.const_vec(0, resolution), math.const_vec(1, resolution)))
    components = values.vector.unstack(resolution.spatial_rank)
    tensors = []
    for dim, component in zip(resolution.spatial.names, components):
        comp_cells = cells.stagger(dim, *extrapolation.valid_outer_faces(dim))
        tensors.append(math.expand(component, comp_cells.resolution))
    return math.stack(tensors, dual(vector=resolution.names))


def resolution_from_staggered_tensor(values: Tensor, extrapolation: Extrapolation):
    any_dim = values.shape.spatial.names[0]
    x_shape = values.shape.after_gather({'vector': any_dim, '~vector': any_dim})
    ext_lower, ext_upper = extrapolation.valid_outer_faces(any_dim)
    delta = int(ext_lower) + int(ext_upper) - 1
    resolution = x_shape.spatial.with_dim_size(any_dim, x_shape.get_size(any_dim) - delta)
    return resolution


def _sample_function(f, elements: Geometry):
    from phiml.math._functional import get_function_parameters
    try:
        params = get_function_parameters(f)
        dims = elements.shape.get_size('vector')
        names_match = tuple(params.keys())[:dims] == elements.shape.get_item_names('vector')
        num_positional = 0
        has_varargs = False
        for n, p in params.items():
            if p.default is p.empty:
                num_positional += 1
            if p.kind == 2:  # _ParameterKind.VAR_POSITIONAL
                has_varargs = True
        assert num_positional <= dims, f"Cannot sample {f.__name__}({', '.join(tuple(params))}) on physical space {elements.shape.get_item_names('vector')}"
        pass_varargs = has_varargs or names_match or num_positional > 1 or num_positional == dims
        if num_positional > 1 and not has_varargs:
            assert names_match, f"Positional arguments of {f.__name__}({', '.join(tuple(params))}) should match physical space {elements.shape.get_item_names('vector')}"
    except ValueError as err:  # signature not available for all functions
        pass_varargs = False
    if pass_varargs:
        values = math.map_s2b(f)(*elements.center.vector)
    else:
        values = math.map_s2b(f)(elements.center)
    assert isinstance(values, math.Tensor), f"values function must return a Tensor but returned {type(values)}"
    return values


def _get_resolution(resolution: Shape, resolution_: dict, bounds: Box):
    assert 'boundaries' not in resolution_, "'boundaries' is not a valid grid argument. Use 'extrapolation' instead, passing a value or math.extrapolation.Extrapolation object. See https://tum-pbs.github.io/PhiFlow/phi/math/extrapolation.html"
    if isinstance(resolution, int):
        assert not resolution_, "Cannot specify keyword resolution and integer resolution at the same time."
        resolution = spatial(**{dim: resolution for dim in bounds.size.shape.get_item_names('vector')})
    try:
        resolution_ = spatial(**resolution_)
    except AssertionError as err:
        raise ValueError(f"Invalid grid resolution: {', '.join(f'{dim}={size}' for dim, size in resolution_.items())}. Pass an int for all sizes.") from err
    return (resolution or math.EMPTY_SHAPE) & resolution_
