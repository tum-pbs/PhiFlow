import warnings

import numpy

from phi.physics.field import CenteredGrid, StaggeredGrid
from phi.viz.plot import FRONT, RIGHT, TOP

EMPTY_FIGURE = {'data': [{'z': None, 'type': 'heatmap'}]}


def dash_graph_plot(data, settings):
    # type: (object, dict) -> dict
    if data is None:
        return EMPTY_FIGURE

    if isinstance(data, (CenteredGrid, numpy.ndarray)):
        if data.rank == 1:
            return plot(data, settings)
        if data.rank == 2:
            return heatmap(data, settings)
        if data.rank == 3:
            return heatmap(slice_2d(data, settings), settings)

    if isinstance(data, StaggeredGrid):
        component = settings.get('component', 'length')
        if component == 'x':
            return heatmap(data.unstack()[-1], settings)

    warnings.warn('No figure recipe for data %s' % data)
    return EMPTY_FIGURE


def heatmap(data, settings):
    batch = settings.get('batch', 0)
    component = settings.get('component', 'x')  # ToDo
    if isinstance(data, CenteredGrid):
        z = data.data[batch,:,:,0]
        y = numpy.linspace(data.box.get_lower(0), data.box.get_upper(0), data.resolution[0])
        x = numpy.linspace(data.box.get_lower(1), data.box.get_upper(1), data.resolution[1])
        return {'data': [{'x': x, 'y': y, 'z': z, 'type': 'heatmap'}]}
    elif isinstance(data, numpy.ndarray):
        return {'data': [{'z': data, 'type': 'heatmap'}]}
    else:
        raise ValueError('Unsupported type for heatmap: %s' % type(data))


def slice_2d(field3d, settings):
    if isinstance(field3d, numpy.ndarray):
        field3d = CenteredGrid(field3d)
    assert isinstance(field3d, CenteredGrid) and field3d.rank == 3
    depth = settings.get('depth', 0)
    projection = settings.get('projection', FRONT)

    if projection == FRONT:
        # Remove Y axis
        data = field3d.data[:, :, min(depth, field3d.resolution[1]), :, :]
        field2d = CenteredGrid(data, box=field3d.box.without_axis(1))
    elif projection == RIGHT:
        # Remove X axis
        data = field3d.data[:, :, min(depth, field3d.resolution[2]), :, :]
        data = numpy.transpose(data, axes=(0, 2, 1, 3))
        field2d = CenteredGrid(data, box=field3d.box.without_axis(2))
    elif projection == TOP:
        # Remove Z axis
        data = field3d.data[:, min(depth, field3d.resolution[0]), :, :, :]
        field2d = CenteredGrid(data, box=field3d.box.without_axis(0))
    else:
        raise ValueError('Unknown projection: %s' % projection)
    return field2d


def plot(field1d, settings):
    batch = settings.get('batch', 0)
    component = settings.get('component', 'x')  # ToDo
    if isinstance(field1d, CenteredGrid):
        x = numpy.linspace(field1d.box.lower, field1d.box.upper, field1d.resolution[0])
        data = field1d.data[min(field1d.resolution[0], batch), :, :]
        return {'data': [{'mode': 'markers+lines', 'type': 'scatter', 'x': x, 'y': data} for i in range(data.shape[-1])]}
    elif isinstance(field1d, numpy.ndarray):
        data = field1d[min(field1d.shape[0], batch), :, :]
        return {'data': [{'mode': 'markers+lines', 'type': 'scatter', 'y': data} for i in range(data.shape[-1])]}
