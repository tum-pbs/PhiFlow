import warnings

import numpy as np
import plotly.figure_factory as plotly_figures

from phi.math import GLOBAL_AXIS_ORDER as physics_config
from phi.field import CenteredGrid, StaggeredGrid, PointCloud
from .colormaps import COLORMAPS
from ... import math
from ...geom import Box
from .viewsettings import FRONT, RIGHT, TOP

EMPTY_FIGURE = {'data': [{'z': None, 'type': 'heatmap'}]}


def dash_graph_plot(data, settings: dict) -> dict:
    if data is None:
        return EMPTY_FIGURE
    try:
        if isinstance(data, np.ndarray):
            data = math.tensor(data, convert=False)

        if isinstance(data, math.Tensor):
            data = CenteredGrid(data, Box(0, math.tensor(data.shape, 'vector')))

        if isinstance(data, (CenteredGrid, StaggeredGrid)):
            component = settings.get('component', 'x')
            if data.spatial_rank == 1:
                return plot(data, settings)
            if data.spatial_rank == 2:
                if component == 'vec2' and data.shape.channel.volume >= 2:
                    return vector_field(data, settings)
                else:
                    return heatmap(data, settings)
            if data.spatial_rank == 3:
                if component == 'vec2' and data.shape.channel.volum >= 2:
                    return vector_field(slice_2d(data, settings), settings)
                else:
                    return heatmap(slice_2d(data, settings), settings)

        if isinstance(data, PointCloud):
            return cloud_plot(data, settings)

        warnings.warn('No figure recipe for data %s' % data)
    except BaseException as err:
        print(f"Error during plotting: {err}")
    return EMPTY_FIGURE


def get_color_interpolation(val, cm_arr):
    """
    Weighted average between point smaller and larger than it

    Args:
      val: 
      cm_arr: 

    Returns:

    """
    if 0 in cm_arr[:, 0]-val:
        center = cm_arr[cm_arr[:, 0] == val][-1]
    else:
        offset_positions = cm_arr[:, 0] - val
        color1 = cm_arr[np.argmax(offset_positions[offset_positions < 0])]  # largest value smaller than val
        color2 = cm_arr[np.argmin(offset_positions[offset_positions > 0])]  # smallest value larger than val
        if color1[0] == color2[0]:
            center = color1
        else:
            x = (val-color1[0]) / (color2[0]-color1[0])  # weight of row2
            center = color1 * (1-x) + color2 * x
    center[0] = val
    return center


def get_div_map(zmin, zmax, equal_scale=False, colormap=None):
    """
    

    Args:
      colormap(list or array, optional): colormap defined as list of [fraction_val, red_frac, green_frac, blue_frac] (Default value = None)
      zmin: 
      zmax: 
      equal_scale:  (Default value = False)

    Returns:

    """
    colormap = COLORMAPS[colormap]
    # Ensure slicing
    cm_arr = np.array(colormap).astype(np.float64)
    # Centeral color
    if 0.5 not in cm_arr[:, 0]:
        central_color = get_color_interpolation(0.5, cm_arr)[1:]
    else:
        central_color = cm_arr[cm_arr[:, 0] == 0.5][-1][1:]
    # Return base
    if zmin == zmax:
        central_color = np.round(central_color).astype(np.int32)
        return [(0, "rgb({},{},{})".format(*central_color)), (1, "rgb({},{},{})".format(*central_color))]
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


def heatmap(field, settings):
    assert isinstance(field, (StaggeredGrid, CenteredGrid))
    assert field.spatial_rank == 2
    batch = settings.get('batch', 0)
    component = settings.get('component', 'x')

    if isinstance(field, StaggeredGrid):
        if component == 'x':
            field = field.unstack()[physics_config.x]
        elif component == 'y':
            field = field.unstack()[physics_config.y]
        elif component == 'z':
            return EMPTY_FIGURE
        elif component == 'length':
            field = field.at_centers()
        else:
            raise ValueError(component)
    z = field.values
    if len(z.shape.batch) > 0:
        z = z.dimension(z.shape.batch.names[0])[batch]
    z = reduce_component(z, component)
    z = z.numpy()
    points = field.points
    if physics_config.is_x_first:
        z = np.transpose(z)
        x = points.vector[0].y[0].numpy()
        y = points.vector[1].x[0].numpy()
    else:
        x = points.vector[1].y[0].numpy()
        y = points.vector[0].x[0].numpy()
    if settings.get('slow_colorbar', False):
        z_min, z_max = settings['minmax']
    else:
        z_min, z_max = np.min(z), np.max(z)
    color_scale = get_div_map(z_min, z_max, equal_scale=True, colormap=settings.get('colormap', None))
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
            field3d = field3d.unstack()[{'z': physics_config.z, 'y': physics_config.y, 'x': physics_config.x}[component] % 3]
        else:
            field3d = field3d.at_centers()
    assert isinstance(field3d, CenteredGrid) and field3d.spatial_rank == 3
    depth = settings.get('depth', 0)
    projection = settings.get('projection', FRONT)

    removed_axis = {FRONT: physics_config.y, RIGHT: physics_config.x, TOP: physics_config.z}[projection] % 3

    data = field3d.values[(slice(None),) + tuple([min(depth, field3d.resolution[i]) if i == removed_axis else slice(None) for i in range(3)]) + (slice(None),)]
    if projection == RIGHT and not physics_config.is_x_first:
        data = np.transpose(data, axes=(0, 2, 1, 3))

    return CenteredGrid(data, box=field3d.box.without_axis(removed_axis))


def plot(field1d, settings):
    assert isinstance(field1d, (CenteredGrid, StaggeredGrid))
    batch = settings.get('batch', 0)
    component = settings.get('component', 'x')
    if isinstance(field1d, StaggeredGrid):
        field1d = field1d.unstack()[0]
    assert isinstance(field1d, CenteredGrid)
    x = field1d.points.vector[0].native()
    data = field1d.values
    if 'batch' in field1d.shape:
        data = data.batch[min(field1d.resolution[0], batch)]
    data = reduce_component(data, component)
    data = data.native()
    return {'data': [{'mode': 'markers+lines', 'type': 'scatter', 'x': x, 'y': data}]}


def cloud_plot(cloud: PointCloud, settings: dict) -> dict:
    """
    Generates Plotly figure dict for the given PointCloud object.

    Args:
        settings: plot settings
        cloud: Single 2D PointCloud which should get plotted.

    Returns:
        Plotly figure dict with the data from the PointCloud.
    """
    points = cloud.points
    points = math.join_dimensions(points, points.shape.batch.without('points'), 'batch').batch[settings.get('batch', 0)]
    x, y = points.vector.unstack_spatial('x,y', to_numpy=True)
    color = cloud.color.points.unstack(len(x), to_python=True)
    if cloud.bounds:
        lower = cloud.bounds.lower.vector.unstack_spatial('x,y', to_python=True)
        upper = cloud.bounds.upper.vector.unstack_spatial('x,y', to_python=True)
    else:
        lower = [np.min(x), np.min(y)]
        upper = [np.max(x), np.max(y)]
    radius = cloud.elements.bounding_radius() * settings['figsize'] / (upper[1] - lower[1])
    radius = math.maximum(radius, 2)
    plot_dict = {
        'data':
            [{
                'mode': 'markers',
                'type': 'scatter',
                'x': x,
                'y': y,
                'marker':
                    {
                        'color': color,
                        'size': (2 * radius).points.optional_unstack(to_python=True),
                        'sizemode': 'diameter',
                    },
            }],
        'layout':
            {
                'xaxis': {'range': [lower[0], upper[0]]},
                'yaxis': {'range': [lower[1], upper[1]]}
            }
    }
    return plot_dict


def reduce_component(tensor, component):
    if tensor.shape.channel.rank == 0:
        return tensor
    assert tensor.shape.channel.rank == 1
    clen = tensor.shape.channel.volume
    if clen == 1:
        return tensor[..., 0]
    if component == 'x':
        return tensor[..., physics_config.x]
    if component == 'y':
        return tensor[..., physics_config.y]
    if component == 'z':
        if clen >= 3:
            return tensor[..., physics_config.z]
        else:
            return math.zeros_like(tensor[..., 0])
    if component == 'length':
        return math.vec_abs(tensor)
    if component == 'vec2':
        return tensor[..., (physics_config.y, physics_config.x)]


def vector_field(field2d, settings):
    assert isinstance(field2d, (CenteredGrid, StaggeredGrid))
    if isinstance(field2d, StaggeredGrid):
        field2d = field2d.at_centers()
    assert isinstance(field2d, CenteredGrid)
    assert field2d.spatial_rank == 2

    batch = settings.get('batch', 0)
    batch = min(batch, field2d.values.shape.batch.volume)

    arrow_origin = settings.get('arrow_origin', 'tip')
    assert arrow_origin in ('base', 'center', 'tip')
    max_resolution = settings.get('max_arrow_resolution', 40)
    max_arrows = settings.get('max_arrows', 2000)
    min_arrow_length = settings.get('min_arrow_length', 0.005) * math.max(field2d.box.size)
    draw_full_arrows = settings.get('draw_full_arrows', False)

    x, y = field2d.points.vector.unstack_spatial('x,y', to_numpy=True)
    data = math.join_dimensions(field2d.values, field2d.shape.batch, 'batch').batch[batch]
    data_x, data_y = data.vector.unstack_spatial('x,y', to_numpy=True)

    while np.prod(x.shape) > max_resolution ** 2:
        y = y[::2, ::2]
        x = x[::2, ::2]
        data_y = data_y[::2, ::2]
        data_x = data_x[::2, ::2]

    y = y.flatten()
    x = x.flatten()
    data_y = data_y.flatten()
    data_x = data_x.flatten()

    if max_arrows is not None or min_arrow_length > 0:
        length = np.sqrt(data_y**2 + data_x**2)
        keep_indices = np.argsort(length)
        keep_indices = keep_indices[length[keep_indices] > min_arrow_length]
        if len(keep_indices) > max_arrows:
            keep_indices = keep_indices[-max_arrows:]
        y = y[keep_indices]
        x = x[keep_indices]
        data_y = data_y[keep_indices]
        data_x = data_x[keep_indices]

    if len(x) == 0:
        return EMPTY_FIGURE

    if arrow_origin == 'tip':
        x -= data_x
        y -= data_y
    elif arrow_origin == 'center':
        x -= 0.5 * data_x
        y -= 0.5 * data_y

    lower = field2d.bounds.lower.vector.unstack_spatial('x,y', to_python=True)
    upper = field2d.bounds.upper.vector.unstack_spatial('x,y', to_python=True)
    x_range = [lower[0], upper[0]]
    y_range = [lower[1], upper[1]]

    if draw_full_arrows:
        result = plotly_figures.create_quiver(x, y, data_x, data_y, scale=1.0)  # 7 points per arrow
        result.update_xaxes(range=x_range)
        result.update_yaxes(range=y_range)
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
                'xaxis': {'range': x_range},
                'yaxis': {'range': y_range},
            }
        }
