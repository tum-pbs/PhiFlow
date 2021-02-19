import numpy as np

from phi import math, geom
from ._field import SampledField
from ._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor
from ..math._tensors import NativeTensor


def write(field: SampledField, file: str):
    if isinstance(field, StaggeredGrid):
        data = field.staggered_tensor().numpy()
    else:
        data = field.values.numpy()
    dim_names = field.values.shape.names
    if isinstance(field, Grid):
        lower = field.box.lower.numpy()
        upper = field.box.upper.numpy()
        extrap = field.extrapolation.to_dict()
        np.savez_compressed(file, dim_names=dim_names, dim_types=field.values.shape.types, field_type=type(field).__name__, lower=lower, upper=upper, extrapolation=extrap, data=data)
    else:
        raise NotImplementedError(f"{type(field)} not implemented. Only Grid allowed.")


def read(file: str, convert_to_backend=True) -> SampledField:
    stored = np.load(file, allow_pickle=True)
    ftype = stored['field_type']
    implemented_types = ('CenteredGrid', 'StaggeredGrid')
    if ftype in implemented_types:
        data = stored['data']
        data = NativeTensor(data, math.Shape(data.shape, stored['dim_names'], stored['dim_types']))
        if convert_to_backend:
            data = math.tensor(data, convert=convert_to_backend)
        lower = math.tensor(stored['lower'])
        upper = math.tensor(stored['upper'])
        extrapolation = math.extrapolation.from_dict(stored['extrapolation'][()])
        if ftype == 'CenteredGrid':
            return CenteredGrid(data, geom.Box(lower, upper), extrapolation)
        elif ftype == 'StaggeredGrid':
            data_ = unstack_staggered_tensor(data)
            return StaggeredGrid(data_, geom.Box(lower, upper), extrapolation)
    raise NotImplementedError(f"{ftype} not implemented ({implemented_types})")
