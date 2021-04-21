from functools import wraps, partial
from numbers import Number
from typing import Callable

from phi import geom
from phi import math
from phi.geom import Box, Geometry
from phi.math import extrapolate_valid_values, DType
from ._field import Field, SampledField, SampledFieldType, unstack
from ._grid import CenteredGrid, Grid, StaggeredGrid, GridType
from ._mask import HardGeometryMask
from ._point_cloud import PointCloud
from ..math.backend import Backend


def laplace(field: Grid, axes=None):
    result = field._op1(lambda tensor: math.laplace(tensor, dx=field.dx, padding=field.extrapolation, dims=axes))
    return result


def spatial_gradient(field: CenteredGrid, type: type = CenteredGrid, stack_dim='vector'):
    """
    Finite difference spatial_gradient.

    This function can operate in two modes:

    * `type=CenteredGrid` approximates the spatial_gradient at cell centers using central differences
    * `type=StaggeredGrid` computes the spatial_gradient at face centers of neighbouring cells

    Args:
        field: centered grid of any number of dimensions (scalar field, vector field, tensor field)
        type: either `CenteredGrid` or `StaggeredGrid`
        stack_dim: name of dimension to be added. This dimension lists the spatial_gradient w.r.t. the spatial dimensions.
            The `field` must not have a dimension of the same name.

    Returns:
        spatial_gradient field of type `type`.

    """
    assert isinstance(field, Grid)
    if type == CenteredGrid:
        values = math.spatial_gradient(field.values, field.dx.vector.as_channel(name=stack_dim), difference='central', padding=field.extrapolation, stack_dim=stack_dim)
        return CenteredGrid(values, field.bounds, field.extrapolation.spatial_gradient())
    elif type == StaggeredGrid:
        assert stack_dim == 'vector'
        return stagger(field, lambda lower, upper: (upper - lower) / field.dx, field.extrapolation.spatial_gradient())
    raise NotImplementedError(f"{type(field)} not supported. Only CenteredGrid and StaggeredGrid allowed.")


def shift(grid: CenteredGrid, offsets: tuple, stack_dim='shift'):
    """
    Wraps :func:`math.shift` for CenteredGrid.

    Args:
      grid: CenteredGrid: 
      offsets: tuple: 
      stack_dim:  (Default value = 'shift')

    Returns:

    """
    data = math.shift(grid.values, offsets, padding=grid.extrapolation, stack_dim=stack_dim)
    return [CenteredGrid(data[i], grid.box, grid.extrapolation) for i in range(len(offsets))]


def stagger(field: CenteredGrid,
            face_function: Callable,
            extrapolation: math.extrapolation.Extrapolation,
            type: type = StaggeredGrid):
    """
    Creates a new grid by evaluating `face_function` given two neighbouring cells.
    One layer of missing cells is inferred from the extrapolation.
    
    This method returns a Field of type `type` which must be either StaggeredGrid or CenteredGrid.
    When returning a StaggeredGrid, the new values are sampled at the faces of neighbouring cells.
    When returning a CenteredGrid, the new grid has the same resolution as `field`.

    Args:
      field: centered grid
      face_function: function mapping (value1: Tensor, value2: Tensor) -> center_value: Tensor
      extrapolation: extrapolation mode of the returned grid. Has no effect on the values.
      type: one of (StaggeredGrid, CenteredGrid)
      field: CenteredGrid: 
      face_function: Callable:
      extrapolation: math.extrapolation.Extrapolation: 
      type: type:  (Default value = StaggeredGrid)

    Returns:
      grid of type matching the `type` argument

    """
    all_lower = []
    all_upper = []
    if type == StaggeredGrid:
        for dim in field.shape.spatial.names:
            all_upper.append(math.pad(field.values, {dim: (0, 1)}, field.extrapolation))
            all_lower.append(math.pad(field.values, {dim: (1, 0)}, field.extrapolation))
        all_upper = math.channel_stack(all_upper, 'vector')
        all_lower = math.channel_stack(all_lower, 'vector')
        values = face_function(all_lower, all_upper)
        return StaggeredGrid(values, field.bounds, extrapolation)
    elif type == CenteredGrid:
        left, right = math.shift(field.values, (-1, 1), padding=field.extrapolation, stack_dim='vector')
        values = face_function(left, right)
        return CenteredGrid(values, field.bounds, extrapolation)
    else:
        raise ValueError(type)


def divergence(field: Grid) -> CenteredGrid:
    """
    Computes the divergence of a grid using finite differences.

    This function can operate in two modes depending on the type of `field`:

    * `CenteredGrid` approximates the divergence at cell centers using central differences
    * `StaggeredGrid` exactly computes the divergence at cell centers

    Args:
        field: vector field as `CenteredGrid` or `StaggeredGrid`

    Returns:
        Divergence field as `CenteredGrid`
    """
    if isinstance(field, StaggeredGrid):
        components = []
        for i, dim in enumerate(field.shape.spatial.names):
            div_dim = math.spatial_gradient(field.values.vector[i], dx=field.dx[i], difference='forward', padding=None, dims=[dim]).gradient[0]
            components.append(div_dim)
        data = math.sum(components, 0)
        return CenteredGrid(data, field.box, field.extrapolation.spatial_gradient())
    elif isinstance(field, CenteredGrid):
        left, right = shift(field, (-1, 1), stack_dim='div_')
        grad = (right - left) / (field.dx * 2)
        components = [grad.vector[i].div_[i] for i in range(grad.div_.size)]
        result = sum(components)
        return result
    else:
        raise NotImplementedError(f"{type(field)} not supported. Only StaggeredGrid allowed.")


def curl(field: Grid, type: type = CenteredGrid):
    assert field.spatial_rank in (2, 3), "curl is only defined in 2 and 3 spatial dimensions."
    if field.spatial_rank == 2 and type == StaggeredGrid:
        assert isinstance(field, CenteredGrid) and 'vector' not in field.shape, f"2D curl requires scalar field but got {field}"
        grad = math.spatial_gradient(field.values, dx=field.dx, difference='forward', padding=None, stack_dim='vector')
        result = grad.vector.flip() * (1, -1)  # (d/dy, -d/dx)
        bounds = Box(field.bounds.lower + 0.5 * field.dx, field.bounds.upper - 0.5 * field.dx)  # lose 1 cell per dimension
        return StaggeredGrid(result, bounds, field.extrapolation.spatial_gradient())
    raise NotImplementedError()


def fourier_laplace(grid: GridType, times=1) -> GridType:
    """ See `phi.math.fourier_laplace()` """
    extrapolation = grid.extrapolation.spatial_gradient().spatial_gradient()
    return grid.with_(values=math.fourier_laplace(grid.values, dx=grid.dx, times=times), extrapolation=extrapolation)


def fourier_poisson(grid: GridType, extrapolation: math.Extrapolation = None, times=1) -> GridType:
    """ See `phi.math.fourier_poisson()` """
    return grid.with_(values=math.fourier_poisson(grid.values, dx=grid.dx, times=times), extrapolation=extrapolation)


def native_call(f, *inputs, channels_last=None, channel_dim='vector', extrapolation=None) -> SampledField or math.Tensor:
    """
    Similar to `phi.math.native_call()`.

    Args:
        f: Function to be called on native tensors of `inputs.values`.
            The function output must have the same dimension layout as the inputs and the batch size must be identical.
        *inputs: `SampledField` or `phi.math.Tensor` instances.
        extrapolation: (Optional) Extrapolation of the output field. If `None`, uses the extrapolation of the first input field.

    Returns:
        `SampledField` matching the first `SampledField` in `inputs`.
    """
    input_tensors = [i.values if isinstance(i, SampledField) else math.tensor(i) for i in inputs]
    result = math.native_call(f, *input_tensors, channels_last=channels_last, channel_dim=channel_dim)
    for i in inputs:
        if isinstance(i, SampledField):
            return i.with_(values=result, extrapolation=extrapolation)
    return result


def minimize(function, x0: Grid, solve_params: math.Solve, callback: Callable = None) -> Grid:
    data_function = _operate_on_values(function, x0)
    try:
        return x0.with_(values=math.minimize(data_function, x0.values, solve_params=solve_params, callback=callback))
    except math.ConvergenceException as exc:
        raise type(exc)(exc.solve, x0, x0.with_(values=exc.x), exc.msg)


def solve(function, y: Grid, x0: Grid, solve_params: math.Solve, constants: tuple or list = (), callback=None):
    if callback is not None:
        def field_callback(x):
            x = x0.with_(values=x)
            callback(x)
    else:
        field_callback = None
    if isinstance(function, LinearFieldFunction):
        value_function = function.value_function(x0)
    else:
        value_function = _operate_on_values(function, x0)
    constants = [c.values if isinstance(c, SampledField) else c for c in constants]
    assert all(isinstance(c, math.Tensor) for c in constants)
    x = math.solve(value_function, y.values, x0.values, solve_params=solve_params, constants=constants, callback=field_callback)
    return x0.with_(values=x)


def _operate_on_values(field_function, *proto_fields):
    """
    Constructs a wrapper function operating on field values from a function operating on fields.
    The wrapper function assembles fields and calls `field_function`.

    This is useful when passing functions to a `phi.math` operation, e.g. `phi.math.solve()`.

    Args:
        field_function: Function whose arguments are fields
        *proto_fields: To specify non-value properties of the fields.

    Returns:
        Wrapper for `field_function` that takes the field values of as input and returns the field values of the result.
    """
    @wraps(field_function)
    def wrapper(*field_data):
        fields = [proto.with_(values=data) for data, proto in zip(field_data, proto_fields)]
        result = field_function(*fields)
        if isinstance(result, math.Tensor):
            return result
        elif isinstance(result, SampledField):
            return result.values
        elif isinstance(result, (tuple, list)):
            return [r.values if isinstance(r, SampledField) else r for r in result]
        else:
            raise ValueError(f"function must return an instance of SampledField or Tensor but returned {result}")
    return wrapper


def linear_function(f, jit_compile=True):
    """ Equivalent to `phi.math.linear_function()` for field functions. """
    return LinearFieldFunction(f, jit_compile)


class LinearFieldFunction:

    def __init__(self, f: Callable, jit_compile):
        self.f = f
        self.jit_compile = jit_compile
        self._tensor_function = None
        # self.input_fields = []
        # self.output: tuple or list or SampledField = None

    # def __call__(self, *args, **kwargs):
    #     """ *args are Tensors """
    #     assert not kwargs
    #     tensor_function = self.value_function(*args)
    #     tensors = [field.values for field in args]
    #     result_tensors = tensor_function(*tensors)
    #     if isinstance(self.output, (tuple, list)):
    #         assert isinstance(result_tensors, (tuple, list)), result_tensors
    #         return [f.with_(values=t) if isinstance(f, SampledField) else t for f, t in zip(self.output, result_tensors)]
    #     else:
    #         if isinstance(result_tensors, (tuple, list)):
    #             result_tensors = result_tensors[0]
    #         return self.output.with_(values=result_tensors)

    def value_function(self, *proto_fields):
        # self.input_fields = proto_fields
        if self._tensor_function is None:
            def tensor_function(*tensors):
                fields = [field.with_(values=t) for field, t in zip(proto_fields, tensors)]
                output = self.f(*fields)
                if isinstance(output, (tuple, list)):
                    return [field.values if isinstance(field, SampledField) else math.tensor(field) for field in output]
                else:
                    return output.values if isinstance(output, SampledField) else math.tensor(output)
            self._tensor_function = math.linear_function(tensor_function, jit_compile=self.jit_compile)
        return self._tensor_function


def jit_compile(f: Callable):
    """
    Wrapper for `phi.math.jit_compile()` where `f` is a function operating on fields instead of tensors.

    Here, the arguments and output of `f` should be instances of `Field`.
    """
    wrapper, _, _, _ = _tensor_wrapper(f, lambda tensor_function: math.jit_compile(tensor_function))
    return wrapper


def _tensor_wrapper(f: Callable, create_tensor_function: Callable):
    """
    Wrapper for `phi.math.jit_compile()` where `f` is a function operating on fields instead of tensors.

    Here, the arguments and output of `f` should be instances of `Field`.
    """
    INPUT_FIELDS = []
    OUTPUT_FIELDS = []

    def tensor_function(*tensors):
        fields = [field.with_(values=t) for field, t in zip(INPUT_FIELDS, tensors)]
        result = f(*fields)
        results = [result] if not isinstance(result, (tuple, list)) else result
        OUTPUT_FIELDS.clear()
        OUTPUT_FIELDS.extend(results)
        result_tensors = [field.values if isinstance(field, SampledField) else math.tensor(field) for field in results]
        return result_tensors

    tensor_trace = create_tensor_function(tensor_function)

    def wrapper(*fields):
        INPUT_FIELDS.clear()
        INPUT_FIELDS.extend(fields)
        tensors = [field.values for field in fields]
        result_tensors = tensor_trace(*tensors)
        result_tensors = [result_tensors] if not isinstance(result_tensors, (tuple, list)) else result_tensors
        result = [f.with_(values=t) if isinstance(f, SampledField) else t for f, t in zip(OUTPUT_FIELDS, result_tensors)]
        return result[0] if len(result) == 1 else result

    return wrapper, tensor_trace, INPUT_FIELDS, OUTPUT_FIELDS


def functional_gradient(f: Callable, wrt: tuple or list = (0,), get_output=False) -> Callable:
    """
    Wrapper for `phi.math.functional_gradient()` where `f` is a function operating on fields instead of tensors.

    Here, the arguments of `f` should be instances of `Field`.
    `f` returns a scalar tensor and optionally auxiliary fields.
    """
    INPUT_FIELDS = []
    OUTPUT_FIELDS = []

    def tensor_function(*tensors):
        fields = [field.with_(values=t) for field, t in zip(INPUT_FIELDS, tensors)]
        result = f(*fields)
        results = [result] if not isinstance(result, (tuple, list)) else result
        assert isinstance(results[0], math.Tensor)
        OUTPUT_FIELDS.clear()
        OUTPUT_FIELDS.extend(results)
        result_tensors = [r.values if isinstance(r, Field) else r for r in results]
        return result_tensors

    tensor_gradient = math.functional_gradient(tensor_function, wrt=wrt, get_output=get_output)

    def wrapper(*fields):
        INPUT_FIELDS.clear()
        INPUT_FIELDS.extend(fields)
        tensors = [field.values for field in fields]
        result_tensors = tuple(tensor_gradient(*tensors))
        proto_fields = []
        if get_output:
            proto_fields.extend(OUTPUT_FIELDS)
        proto_fields.extend([t for i, t in enumerate(INPUT_FIELDS) if i in wrt])
        result = [field.with_(values=t) if isinstance(field, Field) else t for field, t in zip(proto_fields, result_tensors)]
        return result

    return wrapper


def convert(field: SampledField, backend: Backend = None, use_dlpack=True):
    if isinstance(field, Grid):
        return field.with_(values=math.convert(field.values, backend, use_dlpack=use_dlpack))
    elif isinstance(field, PointCloud):
        e_char = field.elements._characteristics_()
        elements = field.elements._with_(**{a: math.convert(v, backend, use_dlpack=use_dlpack) for a, v in e_char.items()})
        return field.with_(elements=elements, values=math.convert(field.values, backend, use_dlpack=use_dlpack))
    else:
        raise ValueError(field)


def data_bounds(field: SampledField):
    data = field.points
    min_vec = math.min(data, dim=data.shape.spatial.names)
    max_vec = math.max(data, dim=data.shape.spatial.names)
    return Box(min_vec, max_vec)


def mean(field: SampledField):
    return math.mean(field.values, field.shape.spatial)


def normalize(field: SampledField, norm: SampledField, epsilon=1e-5):
    data = math.normalize_to(field.values, norm.values, epsilon)
    return field.with_(values=data)


def center_of_mass(density: SampledField):
    assert 'vector' not in density.shape
    return mean(density.points * density) / mean(density)


def pad(grid: Grid, widths: int or tuple or list or dict):
    if isinstance(widths, int):
        widths = {axis: (widths, widths) for axis in grid.shape.spatial.names}
    elif isinstance(widths, (tuple, list)):
        widths = {axis: (width if isinstance(width, (tuple, list)) else (width, width)) for axis, width in zip(grid.shape.spatial.names, widths)}
    else:
        assert isinstance(widths, dict)
    widths_list = [widths[axis] for axis in grid.shape.spatial.names]
    if isinstance(grid, Grid):
        data = math.pad(grid.values, widths, grid.extrapolation)
        w_lower = math.wrap([w[0] for w in widths_list])
        w_upper = math.wrap([w[1] for w in widths_list])
        box = Box(grid.box.lower - w_lower * grid.dx, grid.box.upper + w_upper * grid.dx)
        return type(grid)(data, box, grid.extrapolation)
    raise NotImplementedError(f"{type(grid)} not supported. Only Grid instances allowed.")


def downsample2x(grid: Grid) -> GridType:
    if isinstance(grid, CenteredGrid):
        values = math.downsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, grid.bounds, grid.extrapolation)
    elif isinstance(grid, StaggeredGrid):
        values = []
        for dim, centered_grid in zip(grid.shape.spatial.names, unstack(grid, 'vector')):
            odd_discarded = centered_grid.values[{dim: slice(None, None, 2)}]
            others_interpolated = math.downsample2x(odd_discarded, grid.extrapolation, dims=grid.shape.spatial.without(dim))
            values.append(others_interpolated)
        return StaggeredGrid(math.channel_stack(values, 'vector'), grid.bounds, grid.extrapolation)
    else:
        raise ValueError(type(grid))


def upsample2x(grid: GridType) -> GridType:
    if isinstance(grid, CenteredGrid):
        values = math.upsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, grid.bounds, grid.extrapolation)
    elif isinstance(grid, StaggeredGrid):
        raise NotImplementedError()
    else:
        raise ValueError(type(grid))


def concat(*fields: SampledField, dim: str):
    assert all(isinstance(f, SampledField) for f in fields)
    assert all(isinstance(f, type(fields[0])) for f in fields)
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if isinstance(fields[0], Grid):
        values = math.concat([f.values for f in fields], dim=dim)
        return fields[0].with_(values=values)
    elif isinstance(fields[0], PointCloud):
        elements = geom.concat([f.elements for f in fields], dim, sizes=[f.shape.get_size(dim) for f in fields])
        values = math.concat([math.expand(f.values, f.shape.only(dim)) for f in fields], dim)
        colors = math.concat([math.expand(f.color, f.shape.only(dim)) for f in fields], dim)
        return fields[0].with_(elements=elements, values=values, color=colors)
    raise NotImplementedError(type(fields[0]))


def batch_stack(*fields, dim: str):
    assert all(isinstance(f, SampledField) for f in fields), f"All fields must be SampledFields of the same type but got {fields}"
    assert all(isinstance(f, type(fields[0])) for f in fields), f"All fields must be SampledFields of the same type but got {fields}"
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if isinstance(fields[0], Grid):
        values = math.batch_stack([f.values for f in fields], dim)
        return fields[0].with_(values=values)
    elif isinstance(fields[0], PointCloud):
        elements = geom.stack(*[f.elements for f in fields], dim=dim)
        values = math.batch_stack([f.values for f in fields], dim=dim)
        colors = math.batch_stack([f.color for f in fields], dim=dim)
        return fields[0].with_(elements=elements, values=values, color=colors)
    raise NotImplementedError(type(fields[0]))


def channel_stack(*fields, dim: str):
    assert all(isinstance(f, SampledField) for f in fields), f"All fields must be SampledFields of the same type but got {fields}"
    assert all(isinstance(f, type(fields[0])) for f in fields), f"All fields must be SampledFields of the same type but got {fields}"
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if isinstance(fields[0], Grid):
        values = math.channel_stack([f.values for f in fields], dim)
        return fields[0].with_(values=values)
    elif isinstance(fields[0], PointCloud):
        elements = geom.stack(*[f.elements for f in fields], dim=dim)
        values = math.channel_stack([f.values for f in fields], dim=dim)
        colors = math.channel_stack([f.color for f in fields], dim=dim)
        return fields[0].with_(elements=elements, values=values, color=colors)
    raise NotImplementedError(type(fields[0]))


def abs(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.abs)


def sign(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.sign)


def round_(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.round)


def ceil(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.ceil)


def floor(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.floor)


def sqrt(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.sqrt)


def exp(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.exp)


def isfinite(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.isfinite)


def real(field: SampledFieldType) -> SampledFieldType:
    return field._op1(math.real)


def imag(field: SampledFieldType) -> SampledFieldType:
    return field._op1(math.imag)


def sin(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.sin)


def cos(x: SampledFieldType) -> SampledFieldType:
    return x._op1(math.cos)


def cast(x: SampledFieldType, dtype: DType) -> SampledFieldType:
    return x._op1(partial(math.cast, dtype=dtype))


def assert_close(*fields: SampledField or math.Tensor or Number,
                 rel_tolerance: float = 1e-5,
                 abs_tolerance: float = 0):
    """ Raises an AssertionError if the `values` of the given fields are not close. See `phi.math.assert_close()`. """
    f0 = next(filter(lambda t: isinstance(t, SampledField), fields))
    values = [(f >> f0).values if isinstance(f, SampledField) else math.wrap(f) for f in fields]
    math.assert_close(*values, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)


def where(mask: Field or Geometry, field_true: Field, field_false: Field):
    if isinstance(mask, Geometry):
        mask = HardGeometryMask(mask)
    elif isinstance(mask, SampledField):
        field_true = field_true.at(mask)
        field_false = field_false.at(mask)
    elif isinstance(field_true, SampledField):
        mask = mask.at(field_true)
        field_false = field_false.at(field_true)
    elif isinstance(field_false, SampledField):
        mask = mask.at(field_true)
        field_true = field_true.at(mask)
    else:
        raise NotImplementedError('At least one argument must be a SampledField')
    values = mask.values * field_true.values + (1 - mask.values) * field_false.values
    # values = math.where(mask.values, field_true.values, field_false.values)
    return field_true.with_(values=values)


def l1_loss(field: SampledField, batch_norm: bool or str or tuple or list or math.Shape = True):
    """ L1 loss for the unweighted values of the field. See `phi.math.l1_loss()`. """
    return math.l1_loss(field.values, batch_norm=batch_norm)


def l2_loss(field: SampledField, batch_norm: bool or str or tuple or list or math.Shape = True):
    """ L2 loss for the unweighted values of the field. See `phi.math.l2_loss()`. """
    return math.l2_loss(field.values, batch_norm=batch_norm)


def frequency_loss(field: SampledField,
                   n=2,
                   frequency_falloff=100,
                   threshold=1e-5,
                   batch_norm: bool or str or tuple or list or math.Shape = True,
                   ignore_mean=False):
    """ Frequency loss for the unweighted values of the field. See `phi.math.frequency_loss()`. """
    return math.frequency_loss(field.values, n=n, frequency_falloff=frequency_falloff, threshold=threshold, batch_norm=batch_norm, ignore_mean=ignore_mean)


def stop_gradient(field: GridType):
    """ See `phi.math.stop_gradient()` """
    assert isinstance(field, Grid), type(field)
    # if isinstance(field, PointCloud):
    return field._op1(math.stop_gradient)


def vec_abs(field: SampledField):
    """ See `phi.math.vec_abs()` """
    if isinstance(field, StaggeredGrid):
        field = field.at_centers()
    return field.with_(values=math.vec_abs(field.values))


def vec_squared(field: SampledField):
    """ See `phi.math.vec_squared()` """
    if isinstance(field, StaggeredGrid):
        field = field.at_centers()
    return field.with_(values=math.vec_squared(field.values))


def extrapolate_valid(grid: GridType, valid: GridType, distance_cells=1) -> tuple:
    """
    Extrapolates values of `grid` which are marked by nonzero values in `valid` using `phi.math.extrapolate_valid_values().
    If `values` is a StaggeredGrid, its components get extrapolated independently.

    Args:
        grid: Grid holding the values for extrapolation
        valid: Grid (same type as `values`) marking the positions for extrapolation with nonzero values
        distance_cells: Number of extrapolation steps

    Returns:
        grid: Grid with extrapolated values.
        valid: binary Grid marking all valid values after extrapolation.
    """
    assert isinstance(valid, type(grid)), 'Type of valid Grid must match type of grid.'
    if isinstance(grid, CenteredGrid):
        new_values, new_valid = extrapolate_valid_values(grid.values, valid.values, distance_cells)
        return grid.with_(values=new_values), valid.with_(values=new_valid)
    elif isinstance(grid, StaggeredGrid):
        new_values = []
        new_valid = []
        for cgrid, cvalid in zip(unstack(grid, 'vector'), unstack(valid, 'vector')):
            new_tensor, new_mask = extrapolate_valid(cgrid, valid=cvalid, distance_cells=distance_cells)
            new_values.append(new_tensor.values)
            new_valid.append(new_mask.values)
        return grid.with_(values=math.channel_stack(new_values, 'vector')), valid.with_(values=math.channel_stack(new_valid, 'vector'))
    else:
        raise NotImplementedError()


def discretize(grid: Grid, filled_fraction=0.25):
    """ Treats channel dimensions as batch dimensions. """
    import numpy as np
    data = math.reshaped_native(grid.values, [grid.shape.non_spatial, grid.shape.spatial])
    ranked_idx = np.argsort(data, axis=-1)
    filled_idx = ranked_idx[:, int(round(grid.shape.spatial.volume * (1 - filled_fraction))):]
    filled = np.zeros_like(data)
    np.put_along_axis(filled, filled_idx, 1, axis=-1)
    filled_t = math.reshaped_tensor(filled, [grid.shape.non_spatial, grid.shape.spatial])
    return grid.with_(values=filled_t)
