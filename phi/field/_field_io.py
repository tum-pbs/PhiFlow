from typing import Union

import numpy as np

from phi import geom, math
from phiml.math._shape import from_dict
from ._field import Field
from ._grid import unstack_staggered_tensor, CenteredGrid, StaggeredGrid
from ._field_math import stack
from ..math import extrapolation, wrap, tensor, Shape, channel, Tensor, spatial


def write(field: Field, file: Union[str, Tensor]):
    """
    Writes a field to disc using a NumPy file format.
    Depending on `file`, the data may be split up into multiple files.

    All characteristics of the field are serialized so that it can be fully restored using `read()`.

    See Also:
        `read()`

    Args:
        field: Field to be saved.
        file: Single file as `str` or `Tensor` of string type.
            If `file` is a tensor, the dimensions of `field` are matched to the dimensions of `file`.
            Dimensions of `file` that are missing in `field` result in data duplication.
            Dimensions of `field` that are missing in `file` result in larger files.
    """
    if isinstance(file, str):
        write_single_field(field, file)
    elif isinstance(file, Tensor):
        if file.rank == 0:
            write_single_field(field, file.native())
        else:
            dim = file.shape.names[0]
            files = math.unstack(file, dim)
            fields = field.dimension(dim).unstack(file.shape.get_size(dim))
            for field_, file_ in zip(fields, files):
                write(field_, file_)
    else:
        raise ValueError(file)


def write_single_field(field: Field, file: str):
    if field.is_staggered and field.is_grid:
        data = field.staggered_tensor().numpy(field.shape.names)
    else:
        data = field.values.numpy(field.shape.names)
    dim_names = field.shape.names
    if field.is_grid:
        lower = field.bounds.lower.numpy()
        upper = field.bounds.upper.numpy()
        bounds_item_names = field.bounds.size.vector.item_names
        extrap = field.extrapolation.to_dict()
        field_type = 'StaggeredGrid' if field.is_staggered else 'CenteredGrid'
        np.savez_compressed(file,
                            dim_names=dim_names,
                            dim_types=field.shape.dim_types,
                            dim_item_names=np.asarray(field.shape.item_names, dtype=object),
                            field_type=field_type,
                            lower=lower,
                            upper=upper,
                            bounds_item_names=bounds_item_names,
                            extrapolation=extrap,
                            data=data)
    else:
        raise NotImplementedError(f"{type(field)} not implemented. Only Grid allowed.")


def read(file: Union[str, Tensor], convert_to_backend=True) -> Field:
    """
    Loads a previously saved `Field` from disc.

    See Also:
        `write()`.

    Args:
        file: Single file as `str` or `Tensor` of string type.
            If `file` is a tensor, all contained files are loaded an stacked according to the dimensions of `file`.
        convert_to_backend: Whether to convert the read data to the data format of the default backend, e.g. TensorFlow tensors.

    Returns:
        Loaded `Field`.
    """
    if isinstance(file, str):
        return read_single_field(file, convert_to_backend=convert_to_backend)
    if isinstance(file, Tensor):
        if file.rank == 0:
            return read_single_field(file.native(), convert_to_backend=convert_to_backend)
        else:
            dim = file.shape[0]
            files = math.unstack(file, dim.name)
            fields = [read(file_, convert_to_backend=convert_to_backend) for file_ in files]
            return stack(fields, dim)
    else:
        raise ValueError(file)


def read_single_field(file: str, convert_to_backend=True) -> Field:
    stored = np.load(file, allow_pickle=True)
    ftype = stored['field_type']
    if ftype not in ('CenteredGrid', 'StaggeredGrid'):
        raise NotImplementedError(f"{ftype} not implemented")
    data_arr = stored['data']
    dim_item_names = stored.get('dim_item_names', (None,) * len(data_arr.shape))
    shape_spec = {'names': tuple(stored['dim_names']), 'sizes': data_arr.shape, 'types': tuple(stored['dim_types']), 'item_names': tuple(dim_item_names)}
    data = tensor(data_arr, from_dict(shape_spec), convert=convert_to_backend)
    bounds_item_names = stored.get('bounds_item_names', None)
    if bounds_item_names is None or bounds_item_names.shape == ():  # None or empty array
        bounds_item_names = spatial(data).names
    lower = wrap(stored['lower'], channel(vector=tuple(bounds_item_names))) if stored['lower'].ndim > 0 else wrap(stored['lower'])
    upper = wrap(stored['upper'], channel(vector=tuple(bounds_item_names)))
    extr = extrapolation.from_dict(stored['extrapolation'][()])
    if ftype == 'CenteredGrid':
        return CenteredGrid(data, bounds=geom.Box(lower, upper), extrapolation=extr)
    elif ftype == 'StaggeredGrid':
        data_ = unstack_staggered_tensor(data, extr)
        return StaggeredGrid(data_, bounds=geom.Box(lower, upper), extrapolation=extr)
