import numpy as np

from phi import math, geom
from ._field import SampledField
from ._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor
from ._field_math import stack
from ..math._tensors import NativeTensor


def write(field: SampledField, file: str or math.Tensor):
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
    elif isinstance(file, math.Tensor):
        if file.rank == 0:
            write_single_field(field, file.native())
        else:
            dim = file.shape.names[0]
            files = file.unstack(dim)
            fields = field.dimension(dim).unstack(file.shape.get_size(dim))
            for field_, file_ in zip(fields, files):
                write(field_, file_)
    else:
        raise ValueError(file)


def write_single_field(field: SampledField, file: str):
    if isinstance(field, StaggeredGrid):
        data = field.staggered_tensor().numpy(field.values.shape.names)
    else:
        data = field.values.numpy(field.values.shape.names)
    dim_names = field.values.shape.names
    if isinstance(field, Grid):
        lower = field.box.lower.numpy()
        upper = field.box.upper.numpy()
        extrap = field.extrapolation.to_dict()
        np.savez_compressed(file, dim_names=dim_names, dim_types=field.values.shape.types, field_type=type(field).__name__, lower=lower, upper=upper, extrapolation=extrap, data=data)
    else:
        raise NotImplementedError(f"{type(field)} not implemented. Only Grid allowed.")


def read(file: str or math.Tensor, convert_to_backend=True) -> SampledField:
    """
    Loads a previously saved `SampledField` from disc.

    See Also:
        `write()`.

    Args:
        file: Single file as `str` or `Tensor` of string type.
            If `file` is a tensor, all contained files are loaded an stacked according to the dimensions of `file`.
        convert_to_backend: Whether to convert the read data to the data format of the default backend, e.g. TensorFlow tensors.

    Returns:
        Loaded `SampledField`.
    """
    if isinstance(file, str):
        return read_single_field(file, convert_to_backend=convert_to_backend)
    if isinstance(file, math.Tensor):
        if file.rank == 0:
            return read_single_field(file.native(), convert_to_backend=convert_to_backend)
        else:
            dim = file.shape[0]
            files = file.unstack(dim.name)
            fields = [read(file_, convert_to_backend=convert_to_backend) for file_ in files]
            return stack(fields, dim)
    else:
        raise ValueError(file)


def read_single_field(file: str, convert_to_backend=True) -> SampledField:
    stored = np.load(file, allow_pickle=True)
    ftype = stored['field_type']
    implemented_types = ('CenteredGrid', 'StaggeredGrid')
    if ftype in implemented_types:
        data = stored['data']
        data = NativeTensor(data, math.Shape(data.shape, stored['dim_names'], stored['dim_types']))
        if convert_to_backend:
            data = math.tensor(data, convert=convert_to_backend)
        lower = math.wrap(stored['lower'])
        upper = math.wrap(stored['upper'])
        extrapolation = math.extrapolation.from_dict(stored['extrapolation'][()])
        if ftype == 'CenteredGrid':
            return CenteredGrid(data, bounds=geom.Box(lower, upper), extrapolation=extrapolation)
        elif ftype == 'StaggeredGrid':
            data_ = unstack_staggered_tensor(data, extrapolation)
            return StaggeredGrid(data_, bounds=geom.Box(lower, upper), extrapolation=extrapolation)
    raise NotImplementedError(f"{ftype} not implemented ({implemented_types})")
