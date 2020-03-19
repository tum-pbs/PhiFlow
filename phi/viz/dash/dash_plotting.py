import warnings

import numpy as np

import plotly.figure_factory as plotly_figures

from phi import math
from phi.physics.field import CenteredGrid, StaggeredGrid
from phi.viz.plot import FRONT, RIGHT, TOP
from .colormaps import ORANGE_WHITE_BLUE, BLUE_WHITE_RED, VIRIDIS, CIVIDIS, MAGMA, INFERNO, PLASMA, TWILIGHT

EMPTY_FIGURE = {'data': [{'z': None, 'type': 'heatmap'}]}


def dash_graph_plot(data, settings):
    # type: (object, dict) -> dict
    if data is None:
        return EMPTY_FIGURE

    if isinstance(data, np.ndarray):
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


def get_color_interpolation(val, cm_arr):
    """Weighted average between point smaller and larger than it"""
    if 0 in cm_arr[:, 0]-val:
        center = cm_arr[cm_arr[:, 0] == val][-1]
    else:
        zero_centered = (cm_arr[:, 0]-val)
        row1 = cm_arr[np.argmax(zero_centered[zero_centered < 0])]  # largest value smaller than val
        row2 = cm_arr[np.argmin(zero_centered[zero_centered > 0])]  # smallest value larger than val
        center = row1 * (1-(val-row1[0])/(row2[0]-row1[0])) + row2 * (val-row1[0])/(row2[0]-row1[0])  # Interpolate
    center[0] = val
    return center


def get_div_map(zmin, zmax, equal_scale=False, colormap=ORANGE_WHITE_BLUE):
    """
    :param colormap: colormap defined as list of [fraction_val, red_frac, green_frac, blue_frac]
    :type colormap: list or array
    """
    # Ensure slicing
    cm_arr = np.array(colormap).astype(np.float64)
    # Centeral color
    if 0.5 not in cm_arr[:, 0]:
        central_color = get_color_interpolation(0.5, cm_arr)[1:]
    else:
        central_color = cm_arr[cm_arr[:, 0] == 0.5][-1][1:]
    # Return base
    if zmin == zmax:
        return [("0", "rgb({},{},{})".format(*central_color)), ("1", "rgb({},{},{})".format(*central_color))]
    center = abs(zmin / (zmax - zmin))
    if zmin > 0:
        center = 0
    # Rescaling
    if not equal_scale:
        # Full range, Zero-centered
        neg_flag = cm_arr[:, 0] < 0.5
        pos_flag = cm_arr[:, 0] >= 0.5
        cm_arr[neg_flag, 0] = cm_arr[neg_flag, 0]*2*center  # Scale (0, 0.5) -> (0, center)
        cm_arr[pos_flag, 0] = (cm_arr[pos_flag, 0]-0.5)*2*(1-center)+center  # Scale (0.5, 1) -> (center, 0.5)
        # Drop duplicate zeros. Allow for not center value in original map.
        if zmin == 0:
            cm_arr = cm_arr[np.max(np.arange(len(cm_arr))[cm_arr[:, 0] == 0]):]
    else:
        cm_arr[:, 0] = cm_arr[:, 0]-0.5  # center at zero (-0.5, 0.5)
        # Scale desired range
        if zmax > abs(zmin):
            cm_scale = (1-center)/(np.max(cm_arr[:, 0]))  # scale by plositives
        else:
            cm_scale = center/(np.max(cm_arr[:, 0]))  # scale by negatives
        # Scale the maximum to +1 when centered
        cm_arr[:, 0] *= cm_scale
        cm_arr[:, 0] += center  # center
        # Add zero if it doesn't exist
        if 0 not in cm_arr[:, 0]:
            new_min = get_color_interpolation(0, cm_arr)
            cm_arr = np.vstack([new_min, cm_arr])
        # Add one if it doesn't exist
        if 1 not in cm_arr[:, 0]:
            new_max = get_color_interpolation(1, cm_arr)
            cm_arr = np.vstack([cm_arr, new_max])
        # Compare center
        #new_center = get_color_interpolation(center, cm_arr)
        #if not all(new_center == [center, *central_color]):
        #    print("Failed center comparison.")
        #    print("Center: {}".format(new_center))
        #    print("Center should be: {}".format([center, *central_color]))
        #    assert False
        # Cut to (0, 1)
        cm_arr = cm_arr[cm_arr[:, 0] >= 0]
        cm_arr = cm_arr[cm_arr[:, 0] <= 1]
    cm_str = [("{}".format(val), "rgb({:.0f},{:.0f},{:.0f})".format(*colors)) for val, colors in zip(cm_arr[:, 0], cm_arr[:, 1:])]
    return cm_str


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
    z_min, z_max = settings['minmax']
    color_scale = get_div_map(z_min, z_max, equal_scale=True, colormap=ORANGE_WHITE_BLUE)
    return {'data': [{
        'x': x,
        'y': y,
        'z': z,
        'zauto': 'false',
        'zmin': str(z_min),
        'zmax': str(z_max),
        'type': 'heatmap',
        'colorscale': color_scale,
        # 'colorbar': {'title': settings.units,  }  # TODO: Implement units into PhiFlow
    }]}


def slice_2d(field3d, settings):
    if isinstance(field3d, np.ndarray):
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
        data = np.transpose(data, axes=(0, 2, 1, 3))
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
            return np.zeros_like(tensor[..., 0])
    if component == 'length':
        return np.sqrt(np.sum(tensor**2, axis=-1, keepdims=False))
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

    while np.prod(x.shape) > max_resolution ** 2:
        y = y[::2, ::2]
        x = x[::2, ::2]
        data_y = data_y[::2, ::2]
        data_x = data_x[::2, ::2]

    y = y.flatten()
    x = x.flatten()
    data_y = data_y.flatten()
    data_x = data_x.flatten()

    if max_arrows is not None and len(x) > max_arrows:
        length = np.sqrt(data_y**2 + data_x**2)
        keep_indices = np.argsort(length)[-max_arrows:]
        # size = np.max(field2d.box.size)
        # threshold = size * negligible_threshold
        # keep_condition = (np.abs(data_x) > threshold) | (np.abs(data_y) > threshold)
        # keep_indices = np.where(keep_condition)
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
        lines_y = np.stack([y, y + data_y, [None] * len(x)], -1).flatten()  # 3 points per arrow
        lines_x = np.stack([x, x + data_x, [None] * len(x)], -1).flatten()
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
