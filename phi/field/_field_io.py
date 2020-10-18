import numpy as np

from phi import math, geom
from ._field import SampledField
from ._grid import Grid, CenteredGrid
from ._staggered_grid import StaggeredGrid, unstack_staggered_tensor
from ..math._tensors import NativeTensor


def write(field: SampledField, file: str):
    if isinstance(field, StaggeredGrid):
        data = field.staggered_tensor().numpy()
    else:
        data = field.data.numpy()
    dim_names = field.data.shape.names
    if isinstance(field, Grid):
        lower = field.box.lower.numpy()
        upper = field.box.upper.numpy()
        extrap = field.extrapolation.to_dict()
        np.savez_compressed(file, dim_names=dim_names, dim_types=field.data.shape.types, field_type=type(field).__name__, lower=lower, upper=upper, extrapolation=extrap, data=data)
    else:
        raise NotImplementedError(type(field))


def read(file: str, convert_to_backend=True) -> SampledField:
    stored = np.load(file, allow_pickle=True)
    ftype = stored['field_type']
    if ftype in ('CenteredGrid', 'StaggeredGrid'):
        data = stored['data']
        if convert_to_backend:
            data = math.backend.DYNAMIC_BACKEND.default_backend.as_tensor(data, convert_external=True)
        shape = math.Shape(data.shape, stored['dim_names'], stored['dim_types'])
        data = NativeTensor(data, shape)
        lower = math.tensor(stored['lower'], names='vector')
        upper = math.tensor(stored['upper'], names='vector')
        extrapolation = math.extrapolation.from_dict(stored['extrapolation'][()])
        if ftype == 'CenteredGrid':
            return CenteredGrid(data, geom.box(lower, upper), extrapolation)
        elif ftype == 'StaggeredGrid':
            data_ = unstack_staggered_tensor(data)
            return StaggeredGrid(data_, geom.box(lower, upper), extrapolation)
    raise NotImplementedError()
