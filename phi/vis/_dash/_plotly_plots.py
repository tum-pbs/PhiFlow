import warnings

import numpy
from plotly import graph_objects
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS

from phi import math
from phi.field import SampledField, PointCloud, Grid, StaggeredGrid
from phi.vis._dash.colormaps import COLORMAPS
from phi.vis._plot_util import smooth_uniform_curve


def plot(field: SampledField, title=False, show_color_bar=True, size=(800, 600), same_scale=True, colormap: str = None):
    fig_shape = figure_shape(field)
    if fig_shape.volume > 8:
        warnings.warn(f"Plotting {fig_shape.volume} sub-figures for remaining shape {fig_shape} which may be slow. Use 'select' to avoid drawing all examples in one figure.")
    title = titles(title, fig_shape, no_title=None)
    if fig_shape:  # subplots
        fig = make_subplots(rows=1, cols=fig_shape.volume, subplot_titles=title)
        for i, subfig_index in enumerate(fig_shape.meshgrid()):
            sub_field = field[subfig_index]
            _plot(sub_field, fig, row=1, col=i + 1,
                  size=size, colormap=colormap, show_color_bar=show_color_bar)
    else:
        fig = graph_objects.Figure()
        _plot(field, fig,
              size=size, colormap=colormap, show_color_bar=show_color_bar)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def _plot(field: SampledField,
          fig: graph_objects.Figure,
          size: tuple,
          colormap: str,
          show_color_bar: bool,
          row: int = None, col: int = None,
          ):
    if field.spatial_rank == 1 and isinstance(field, Grid):
        x = field.points.numpy().flatten()
        y = math.reshaped_native(real_values(field), [field.shape.spatial], to_numpy=True)
        fig.add_trace(graph_objects.Scatter(x=x, y=y, mode='lines+markers'), row=row, col=col)
        fig.update_layout(showlegend=False)
    elif field.spatial_rank == 2 and isinstance(field, Grid) and field.shape.channel.volume == 1:  # heatmap
        values = real_values(field).numpy('y,x')
        x = field.points.vector['x'].y[0].numpy()
        y = field.points.vector['y'].x[0].numpy()
        zmin, zmax = numpy.nanmin(values), numpy.nanmax(values)
        if not numpy.isfinite(zmin):
            zmin = 0
        if not numpy.isfinite(zmax):
            zmax = 0
        color_scale = get_div_map(zmin, zmax, equal_scale=True, colormap=colormap)
        # color_bar = graph_objects.heatmap.ColorBar(x=1.15)   , colorbar=color_bar
        fig.add_heatmap(row=row, col=col, x=x, y=y, z=values, zauto=False, zmin=zmin, zmax=zmax, colorscale=color_scale, showscale=show_color_bar)
        fig.update_xaxes(scaleanchor='y', scaleratio=1, constrain='domain')
        fig.update_yaxes(constrain='domain')
    elif field.spatial_rank == 2 and isinstance(field, Grid):  # vector field
        if isinstance(field, StaggeredGrid):
            field = field.at_centers()
        x, y = [d.numpy('x,y') for d in field.points.vector.unstack_spatial('x,y')]
        data_x, data_y = [d.numpy('x,y') for d in real_values(field).vector.unstack_spatial('x,y')]
        lower_x, lower_y = [float(l) for l in field.bounds.lower.vector.unstack_spatial('x,y')]
        upper_x, upper_y = [float(u) for u in field.bounds.upper.vector.unstack_spatial('x,y')]
        x_range = [lower_x, upper_x]
        y_range = [lower_y, upper_y]
        # result = figure_factory.create_quiver(x, y, data_x, data_y, scale=1.0)  # 7 points per arrow
        # result.update_xaxes(range=x_range)
        # result.update_yaxes(range=y_range)
        y = y.flatten()
        x = x.flatten()
        data_y = data_y.flatten()
        data_x = data_x.flatten()
        lines_y = numpy.stack([y, y + data_y, [None] * len(x)], -1).flatten()  # 3 points per arrow
        lines_x = numpy.stack([x, x + data_x, [None] * len(x)], -1).flatten()
        fig.add_scatter(x=lines_x, y=lines_y, mode='lines', row=row, col=col)
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)
        fig.update_layout(showlegend=False)
    elif field.spatial_rank == 2 and isinstance(field, PointCloud):
        x, y = [d.numpy() for d in field.points.vector.unstack_spatial('x,y')]
        color = [str(d) for d in field.color.points.unstack(len(x))]
        if field.bounds:
            lower_x, lower_y = [float(d) for d in field.bounds.lower.vector.unstack_spatial('x,y')]
            upper_x, upper_y = [float(d) for d in field.bounds.upper.vector.unstack_spatial('x,y')]
        else:
            lower_x, lower_y = [numpy.min(x), numpy.min(y)]
            upper_x, upper_y = [numpy.max(x), numpy.max(y)]
        radius = field.elements.bounding_radius() * size[1] / (upper_y - lower_y)
        radius = math.maximum(radius, 2)
        if radius.rank == 0:
            marker_size = 2 * float(radius)
        else:
            marker_size = (2 * radius).unstack(radius.shape.instance.name)
        marker = graph_objects.scatter.Marker(size=marker_size, color=color, sizemode='diameter')
        fig.add_scatter(mode='markers', x=x, y=y, marker=marker, row=row, col=col)
        fig.update_xaxes(range=[lower_x, upper_x])
        fig.update_yaxes(range=[lower_y, upper_y])
        fig.update_layout(showlegend=False)
    else:
        raise NotImplementedError(f"No figure recipe for {field}")


def real_values(field: SampledField):
    return field.values if field.values.dtype.kind != complex else abs(field.values)


def figure_shape(field: SampledField):
    if isinstance(field, PointCloud):
        return field.shape.batch
    else:
        return field.shape.batch


def titles(title: bool or str or tuple or list or math.Tensor, fig_shape: math.Shape, no_title: str = None) -> math.Tensor:
    def get_sub_title(title, index):
        if isinstance(title, str):
            return title
        elif title is True:
            return f"{index} of {fig_shape}"
        else:
            return no_title

    if isinstance(title, (tuple, list)):
        title = math.reshaped_tensor(title, [fig_shape])
    return math.map(get_sub_title, math.tensor(title), math.range_tensor(fig_shape))


def get_div_map(zmin, zmax, equal_scale=False, colormap: str = None):
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
    cm_arr = numpy.array(colormap).astype(numpy.float64)
    # Centeral color
    if 0.5 not in cm_arr[:, 0]:
        central_color = get_color_interpolation(0.5, cm_arr)[1:]
    else:
        central_color = cm_arr[cm_arr[:, 0] == 0.5][-1][1:]
    # Return base
    if zmin == zmax:
        central_color = numpy.round(central_color).astype(numpy.int32)
        return [(0, "rgb({},{},{})".format(*central_color)), (1, "rgb({},{},{})".format(*central_color))]
    center = abs(zmin / (zmax - zmin))
    if zmin > 0:
        center = 0
    # Rescaling
    if not equal_scale:
        # Full range, Zero-centered
        neg_flag = cm_arr[:, 0] < 0.5
        pos_flag = cm_arr[:, 0] >= 0.5
        cm_arr[neg_flag, 0] = cm_arr[neg_flag, 0] * 2 * center  # Scale (0, 0.5) -> (0, center)
        cm_arr[pos_flag, 0] = (cm_arr[pos_flag, 0] - 0.5) * 2 * (1 - center) + center  # Scale (0.5, 1) -> (center, 0.5)
        # Drop duplicate zeros. Allow for not center value in original map.
        if zmin == 0:
            cm_arr = cm_arr[numpy.max(numpy.arange(len(cm_arr))[cm_arr[:, 0] == 0]):]
    else:
        cm_arr[:, 0] = cm_arr[:, 0] - 0.5  # center at zero (-0.5, 0.5)
        # Scale desired range
        if zmax > abs(zmin):
            cm_scale = (1 - center) / (numpy.max(cm_arr[:, 0]))  # scale by plositives
        else:
            cm_scale = center / (numpy.max(cm_arr[:, 0]))  # scale by negatives
        # Scale the maximum to +1 when centered
        cm_arr[:, 0] *= cm_scale
        cm_arr[:, 0] += center  # center
        # Add zero if it doesn't exist
        if 0 not in cm_arr[:, 0]:
            new_min = get_color_interpolation(0, cm_arr)
            cm_arr = numpy.vstack([new_min, cm_arr])
        # Add one if it doesn't exist
        if 1 not in cm_arr[:, 0]:
            new_max = get_color_interpolation(1, cm_arr)
            cm_arr = numpy.vstack([cm_arr, new_max])
        # Compare center
        # new_center = get_color_interpolation(center, cm_arr)
        # if not all(new_center == [center, *central_color]):
        #    print("Failed center comparison.")
        #    print("Center: {}".format(new_center))
        #    print("Center should be: {}".format([center, *central_color]))
        #    assert False
        # Cut to (0, 1)
        cm_arr = cm_arr[cm_arr[:, 0] >= 0]
        cm_arr = cm_arr[cm_arr[:, 0] <= 1]
    cm_arr[:, 1:] = numpy.clip(cm_arr[:, 1:], 0, 255)
    return [[val, "rgb({:.0f},{:.0f},{:.0f})".format(*colors)] for val, colors in zip(cm_arr[:, 0], cm_arr[:, 1:])]


def get_color_interpolation(val, cm_arr):
    """
    Weighted average between point smaller and larger than it

    Args:
      val: 
      cm_arr: 

    Returns:

    """
    if 0 in cm_arr[:, 0] - val:
        center = cm_arr[cm_arr[:, 0] == val][-1]
    else:
        offset_positions = cm_arr[:, 0] - val
        color1 = cm_arr[numpy.argmax(offset_positions[offset_positions < 0])]  # largest value smaller than control
        color2 = cm_arr[numpy.argmin(offset_positions[offset_positions > 0])]  # smallest value larger than control
        if color1[0] == color2[0]:
            center = color1
        else:
            x = (val - color1[0]) / (color2[0] - color1[0])  # weight of row2
            center = color1 * (1 - x) + color2 * x
    center[0] = val
    return center


def split_curve(x, y):
    backtracks = numpy.argwhere(x[1:] < x[:-1])[:, 0] + 1
    if len(backtracks) > 0:
        x = numpy.insert(numpy.array(x, numpy.float), backtracks, numpy.nan)
        y = numpy.insert(numpy.array(y, numpy.float), backtracks, numpy.nan)
    return x, y


def plot_scalars(curves: tuple or list, labels, subplots=True, log_scale='', smooth: int = 1):
    if not curves:
        return graph_objects.Figure()
    if subplots:
        fig = make_subplots(rows=1, cols=len(curves), subplot_titles=labels)
        for col, (label, (x, y)) in enumerate(zip(labels, curves)):
            for trace in _graph(label, x, y, smooth, col):
                fig.add_trace(trace, row=1, col=1 + col)
    else:
        fig = graph_objects.Figure()
        for col, (label, (x, y)) in enumerate(zip(labels, curves)):
            for trace in _graph(label, x, y, smooth, col):
                fig.add_trace(trace)
    fig.update_layout(showlegend=not subplots, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if 'x' in log_scale:
        fig.update_xaxes(type='log')
    if 'y' in log_scale:
        fig.update_yaxes(type='log')
    return fig


def _graph(label: str, x, y, smooth: int, index: int):
    color = DEFAULT_PLOTLY_COLORS[index % len(DEFAULT_PLOTLY_COLORS)]
    x, y = split_curve(x, y)
    if smooth > 1:
        smooth_x, smooth_y = smooth_uniform_curve(x, y, n=smooth)
        transparent_color = f"rgba{color[3:-1]}, 0.4)"
        return [
            graph_objects.Scatter(x=x, y=y, line=graph_objects.scatter.Line(color=transparent_color, width=1), showlegend=False),
            graph_objects.Scatter(x=smooth_x, y=smooth_y, name=label, line=graph_objects.scatter.Line(color=color, width=3), mode='lines')
        ]
    else:
        return [graph_objects.Scatter(x=x, y=y, name=label, line=graph_objects.scatter.Line(color=color))]

