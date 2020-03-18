import warnings

import numpy

import plotly.figure_factory as plotly_figures

from phi import math
from phi.physics.field import CenteredGrid, StaggeredGrid
from phi.viz.plot import FRONT, RIGHT, TOP

EMPTY_FIGURE = {'data': [{'z': None, 'type': 'heatmap'}]}


def dash_graph_plot(data, settings):
    # type: (object, dict) -> dict
    if data is None:
        return EMPTY_FIGURE

    if isinstance(data, numpy.ndarray):
        data = CenteredGrid(data)

    if isinstance(data, (CenteredGrid, StaggeredGrid)):
        component = settings.get('component', 'x')
        if data.rank == 1:
            return plot(data, settings)
        if data.rank == 2:
            if component == 'vec2' and data.component_count >= 2:
                return vector_field(data, settings)
            else:
                return heatmap(data, settings)
        if data.rank == 3:
            if component == 'vec2' and data.component_count >= 2:
                return vector_field(slice_2d(data, settings), settings)
            else:
                return heatmap(slice_2d(data, settings), settings)

    warnings.warn('No figure recipe for data %s' % data)
    return EMPTY_FIGURE


def heatmap(data, settings):
    assert isinstance(data, (StaggeredGrid, CenteredGrid))
    assert data.rank == 2
    batch = settings.get('batch', 0)
    component = settings.get('component', 'x')

    if isinstance(data, StaggeredGrid):
        if component == 'x':
            data = data.unstack()[-1]
        elif component == 'y':
            data = data.unstack()[-2]
        elif component == 'z':
            return EMPTY_FIGURE
        elif component == 'length':
            data = data.at_centers()
        else:
            raise ValueError(component)
    z = data.data[batch, ...]
    z = reduce_component(z, component)
    y = data.points.data[0, :, 0, 0]
    x = data.points.data[0, 0, :, 1]
    zmin, zmax = settings['minmax']
    zmin = zmin[0]
    zmax = zmax[0]
    colorsettings = {'colorscale': [(0, 'rgb(240,240,240)'), (1, 'rgb(210,0,0)')],
                     'zauto': 'false', 'zmin': str(zmin), 'zmax': str(zmax)}
    if zmin < 0:
        center = abs(zmin/(zmax-zmin))
        colorsettings['colorscale'] = [(0, 'rgb(0,0,170)'), (center, 'rgb(240,240,240)'), (1, 'rgb(210,0,0)')]
    return {'data': [{'x': x, 'y': y, 'z': z, 'type': 'heatmap',
                      **colorsettings,
                      #'colorbar': {'title': settings.units,  }  # TODO: Implement units into PhiFlow
           }]}


def slice_2d(field3d, settings):
    if isinstance(field3d, numpy.ndarray):
        field3d = CenteredGrid(field3d)
    if isinstance(field3d, StaggeredGrid):
        component = settings.get('component', 'length')
        if component in ('z', 'y', 'x'):
            field3d = field3d.unstack()[('z', 'y', 'x').index(component)]
        else:
            field3d = field3d.at_centers()
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
    assert isinstance(field1d, (CenteredGrid, StaggeredGrid))
    batch = settings.get('batch', 0)
    component = settings.get('component', 'x')
    if isinstance(field1d, StaggeredGrid):
        field1d = field1d.unstack()[0]
    assert isinstance(field1d, CenteredGrid)
    x = field1d.points.data[0, :, 0]
    data = field1d.data[min(field1d.resolution[0], batch), :, :]
    data = reduce_component(data, component)
    return {'data': [{'mode': 'markers+lines', 'type': 'scatter', 'x': x, 'y': data}]}


def reduce_component(tensor, component):
    clen = tensor.shape[-1]
    if clen == 1:
        return tensor[..., 0]
    if component == 'x':
        return tensor[..., -1]
    if component == 'y':
        return tensor[..., -2]
    if component == 'z':
        if clen >= 3:
            return tensor[..., -3]
        else:
            return numpy.zeros_like(tensor[..., 0])
    if component == 'length':
        return numpy.sqrt(numpy.sum(tensor**2, axis=-1, keepdims=False))
    if component == 'vec2':
        return tensor[..., -2:]


def vector_field(field2d, settings):
    assert isinstance(field2d, (CenteredGrid, StaggeredGrid))
    if isinstance(field2d, StaggeredGrid):
        field2d = field2d.at_centers()
    assert isinstance(field2d, CenteredGrid)
    assert field2d.rank == 2

    batch = settings.get('batch', 0)
    batch = min(batch, field2d.data.shape[0])

    arrow_origin = settings.get('arrow_origin', 'tip')
    assert arrow_origin in ('base', 'center', 'tip')
    max_resolution = settings.get('max_arrow_resolution', 40)
    max_arrows = settings.get('max_arrows', 300)
    draw_full_arrows = settings.get('draw_full_arrows', False)

    y, x = math.unstack(field2d.points.data[0, ..., -2:], axis=-1)
    data_y, data_x = math.unstack(field2d.data[batch, ...], -1)[-2:]

    while numpy.prod(x.shape) > max_resolution ** 2:
        y = y[::2, ::2]
        x = x[::2, ::2]
        data_y = data_y[::2, ::2]
        data_x = data_x[::2, ::2]

    y = y.flatten()
    x = x.flatten()
    data_y = data_y.flatten()
    data_x = data_x.flatten()

    if max_arrows is not None and len(x) > max_arrows:
        length = numpy.sqrt(data_y**2 + data_x**2)
        keep_indices = numpy.argsort(length)[-max_arrows:]
        # size = numpy.max(field2d.box.size)
        # threshold = size * negligible_threshold
        # keep_condition = (numpy.abs(data_x) > threshold) | (numpy.abs(data_y) > threshold)
        # keep_indices = numpy.where(keep_condition)
        y = y[keep_indices]
        x = x[keep_indices]
        data_y = data_y[keep_indices]
        data_x = data_x[keep_indices]

    if arrow_origin == 'tip':
        x -= data_x
        y -= data_y
    elif arrow_origin == 'center':
        x -= 0.5 * data_x
        y -= 0.5 * data_y

    if draw_full_arrows:
        result = plotly_figures.create_quiver(x, y, data_x, data_y, scale=1.0)  # 7 points per arrow
        result.update_xaxes(range=[field2d.box.get_lower(1), field2d.box.get_upper(1)])
        result.update_yaxes(range=[field2d.box.get_lower(0), field2d.box.get_upper(0)])
        return result
    else:
        lines_y = numpy.stack([y, y + data_y, [None] * len(x)], -1).flatten()  # 3 points per arrow
        lines_x = numpy.stack([x, x + data_x, [None] * len(x)], -1).flatten()
        return {
            'data': [
                {
                    'mode': 'lines',
                    'x': lines_x,
                    'y': lines_y,
                    'type': 'scatter',
                }
            ],
            'layout': {
                'xaxis': {'range': [field2d.box.get_lower(1), field2d.box.get_upper(1)]},
                'yaxis': {'range': [field2d.box.get_lower(0), field2d.box.get_upper(0)]},
            }
        }
