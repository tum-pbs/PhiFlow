import numpy as np

from phi.geom import *


def grid(griddef, points, property=None, default_value=0):
    indices = (points - 0.5).astype(np.int)[..., ::-1]
    indices = math.unstack(indices, axis=-1)
    if property is None:
        array = griddef.zeros()
        array[[0] + indices + [0]] = 1
    else:
        array = griddef.zeros(property.shape[-1]) + default_value
        array[[0] + indices + [slice(None)]] += property
    return array


def active_centers(array):
    assert array.shape[-1] == 1
    index_array = []
    for batch in range(array.shape[0]):
        indices = np.argwhere(array[batch, ..., 0] > 0)[:, ::-1]
        index_array.append(indices)
    try:
        index_array = np.stack(index_array)
    except ValueError:
        raise ValueError("all arrays in the batch must have the same number of active cells.")
    return index_array + 0.5
