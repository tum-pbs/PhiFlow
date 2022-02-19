import warnings
from typing import List, Tuple, Any, Dict, Optional

import numpy
import plotly.graph_objs
from plotly import graph_objects
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS

from phi import math
from phi.field import SampledField, PointCloud, Grid, StaggeredGrid, tensor_as_field
from phi.math import instance, Tensor
from phi.vis._dash.colormaps import COLORMAPS
from phi.vis._plot_util import smooth_uniform_curve
from phi.vis._vis_base import PlottingLibrary


class PlotlyPlots(PlottingLibrary):

    def __init__(self):
        self.last_fig: Optional[plotly.graph_objs.Figure] = None

    def create_figure(self,
                      size: tuple,
                      rows: int,
                      cols: int,
                      subplots: Dict[Tuple[int, int], int],
                      titles: Tensor) -> Tuple[Any, Dict[Tuple[int, int], Any]]:
        titles = [titles.rows[r].cols[c].native() for r in range(rows) for c in range(cols)]
        specs = [[{'type': 'xy' if subplots.get((row, col), 0) < 3 else 'surface'} for col in range(cols)] for row in range(rows)]
        fig = self.last_fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, specs=specs)
        return fig, {pos: (pos[0]+1, pos[1]+1) for pos in subplots.keys()}

    def plot(self, data: SampledField, figure, subplot, min_val: float = None, max_val: float = None,
             show_color_bar: bool = True, **plt_args):
        _plot(data, figure, row=subplot[0], col=subplot[1], size=(800, 600), colormap=None, show_color_bar=show_color_bar)

    def show(self, figure=None):
        if figure is None:
            figure = self.last_fig
        if figure is not None:
            figure.show()


PLOTLY = PlotlyPlots()


def plot(fields: SampledField or Tensor or List[SampledField or Tensor],
         title=False, show_color_bar=True, size=(800, 600), same_scale=True, colormap: str = None):
    if not isinstance(fields, (tuple, list)):
        fields = [fields]
    fields = [f if isinstance(f, SampledField) else tensor_as_field(f) for f in fields]
    fig_shape = math.merge_shapes(*[f.shape.batch for f in fields])
    if fig_shape.volume > 8:
        warnings.warn(f"Plotting {fig_shape.volume} sub-figures for remaining shape {fig_shape} which may be slow. Use 'select' to avoid drawing all examples in one figure.")
    title = titles(title, fig_shape, no_title=None)
    if fig_shape:  # subplots
        fig = make_subplots(rows=1, cols=fig_shape.volume, subplot_titles=title)
        for i, subfig_index in enumerate(fig_shape.meshgrid()):
            for field in fields:
                sub_field = field[subfig_index]
                _plot(sub_field, fig, row=1, col=i + 1, size=size, colormap=colormap, show_color_bar=show_color_bar)
    else:
        fig = graph_objects.Figure()
        for field in fields:
            _plot(field, fig, size=size, colormap=colormap, show_color_bar=show_color_bar)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def _plot(field: SampledField,
          fig: graph_objects.Figure,
          size: tuple,
          colormap: str or None,
          show_color_bar: bool,
          row: int = None, col: int = None,
          ):
    subplot = fig.get_subplot(row, col)
    subplot_height = (subplot.yaxis.domain[1] - subplot.yaxis.domain[0]) * size[1]
    if field.spatial_rank == 1 and isinstance(field, Grid):
        x = field.points.vector[0].numpy().flatten()
        channels = field.values.shape.channel
        if channels.rank == 1 and channels.get_item_names(0) is not None:
            for i, name in enumerate(channels.get_item_names(0)):
                y = math.reshaped_native(real_values(field[{channels.name: i}]), [field.shape.spatial], to_numpy=True)
                fig.add_trace(graph_objects.Scatter(x=x, y=y, mode='lines+markers', name=name), row=row, col=col)
            fig.update_layout(showlegend=True)
        else:
            for channel in channels.meshgrid():
                y = math.reshaped_native(real_values(field[channel]), [field.shape.spatial], to_numpy=True)
                fig.add_trace(graph_objects.Scatter(x=x, y=y, mode='lines+markers', name='Multi-channel'), row=row, col=col)
            fig.update_layout(showlegend=False)
    elif field.spatial_rank == 2 and isinstance(field, Grid) and 'vector' not in field.shape:  # heatmap
        values = real_values(field).numpy('y,x')
        x = field.points.vector['x'].y[0].numpy()
        y = field.points.vector['y'].x[0].numpy()
        min_val, max_val = numpy.nanmin(values), numpy.nanmax(values)
        min_val, max_val = min_val if numpy.isfinite(min_val) else 0, max_val if numpy.isfinite(max_val) else 0
        color_scale = get_div_map(min_val, max_val, equal_scale=True, colormap=colormap)
        # color_bar = graph_objects.heatmap.ColorBar(x=1.15)   , colorbar=color_bar
        fig.add_heatmap(row=row, col=col, x=x, y=y, z=values, zauto=False, zmin=min_val, zmax=max_val, colorscale=color_scale, showscale=show_color_bar)
        subplot.xaxis.update(scaleanchor=f'y{subplot.yaxis.plotly_name[5:]}', scaleratio=1, constrain='domain')
        subplot.yaxis.update(constrain='domain')
    elif field.spatial_rank == 2 and isinstance(field, Grid):  # vector field
        if isinstance(field, StaggeredGrid):
            field = field.at_centers()
        x, y = [d.numpy('x,y') for d in field.points.vector.unstack_spatial('x,y')]
        # ToDo Additional channel dims as multiple vectors
        extra_channels = field.shape.channel.without('vector')
        values = math.pack_dims(real_values(field), extra_channels, math.channel('channels'))
        data_x, data_y = [d.numpy('channels,x,y') for d in values.vector.unstack_spatial('x,y')]
        lower_x, lower_y = [float(l) for l in field.bounds.lower.vector.unstack_spatial('x,y')]
        upper_x, upper_y = [float(u) for u in field.bounds.upper.vector.unstack_spatial('x,y')]
        x_range = [lower_x, upper_x]
        y_range = [lower_y, upper_y]
        y = y.flatten()
        x = x.flatten()
        for ch in range(data_x.shape[0]):
            # quiver = figure_factory.create_quiver(x, y, data_x[ch], data_y[ch], scale=1.0)  # 7 points per arrow
            # fig.add_trace(quiver, row=row, col=col)
            data_y_flat = data_y[ch].flatten()
            data_x_flat = data_x[ch].flatten()
            # lines_y = numpy.stack([y, y + data_y_flat, [None] * len(x)], -1).flatten()  # 3 points per arrow
            # lines_x = numpy.stack([x, x + data_x_flat, [None] * len(x)], -1).flatten()
            lines_y = numpy.stack([y - data_y_flat / 2, y + data_y_flat / 2, [None] * len(x)], -1).flatten()  # 3 points per arrow
            lines_x = numpy.stack([x - data_x_flat / 2, x + data_x_flat / 2, [None] * len(x)], -1).flatten()
            name = extra_channels.get_item_names(0)[ch] if extra_channels.rank == 1 and extra_channels.get_item_names(0) is not None else None
            fig.add_scatter(x=lines_x, y=lines_y, mode='lines', row=row, col=col, name=name)
        if data_x.shape[0] == 1:
            fig.update_layout(showlegend=False)
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)
        subplot.xaxis.update(scaleanchor=f'y{subplot.yaxis.plotly_name[5:]}', scaleratio=1, constrain='domain')
        subplot.yaxis.update(constrain='domain')
    elif field.spatial_rank == 3 and isinstance(field, Grid) and field.shape.channel.volume == 1:  # 3D heatmap
        values = real_values(field).numpy('z,y,x')
        x = field.points.vector['x'].numpy('z,y,x')
        y = field.points.vector['y'].numpy('z,y,x')
        z = field.points.vector['z'].numpy('z,y,x')
        min_val, max_val = numpy.nanmin(values), numpy.nanmax(values)
        min_val, max_val = min_val if numpy.isfinite(min_val) else 0, max_val if numpy.isfinite(max_val) else 0
        color_scale = get_div_map(min_val, max_val, equal_scale=True, colormap=colormap)
        fig.add_volume(x=x.flatten(), y=y.flatten(), z=z.flatten(), value=values.flatten(),
                       showscale=show_color_bar, colorscale=color_scale, cmin=min_val, cmax=max_val, cauto=False,
                       isomin=0.1, isomax=0.8,
                       opacity=0.1,  # needs to be small to see through all surfaces
                       surface_count=17,  # needs to be a large number for good volume rendering
                       row=row, col=col)
        fig.update_layout(uirevision=True)
    elif field.spatial_rank == 3 and isinstance(field, Grid):  # 3D vector field
        if isinstance(field, StaggeredGrid):
            field = field.at_centers()
        u = real_values(field).vector['x'].numpy('z,y,x')
        v = real_values(field).vector['y'].numpy('z,y,x')
        w = real_values(field).vector['z'].numpy('z,y,x')
        x = field.points.vector['x'].numpy('z,y,x')
        y = field.points.vector['y'].numpy('z,y,x')
        z = field.points.vector['z'].numpy('z,y,x')
        fig.add_cone(x=x.flatten(), y=y.flatten(), z=z.flatten(), u=u.flatten(), v=v.flatten(), w=w.flatten(),
                     colorscale='Blues',
                     sizemode="absolute", sizeref=1,
                     row=row, col=col)
    elif field.spatial_rank == 2 and isinstance(field, PointCloud):
        x, y = [d.numpy() for d in field.points.vector.unstack_spatial('x,y')]
        if field.color.shape.instance_rank == 0:
            color = str(field.color)
        else:
            color = [str(d) for d in math.unstack(field.color, instance)]
        if field.bounds:
            lower_x, lower_y = [float(d) for d in field.bounds.lower.vector.unstack_spatial('x,y')]
            upper_x, upper_y = [float(d) for d in field.bounds.upper.vector.unstack_spatial('x,y')]
        else:
            lower_x, lower_y = [numpy.min(x), numpy.min(y)]
            upper_x, upper_y = [numpy.max(x), numpy.max(y)]
        radius = field.elements.bounding_radius() * subplot_height / (upper_y - lower_y)
        radius = math.maximum(radius, 2)
        marker_size = 1.4142 * 2 * (float(radius) if radius.rank == 0 else radius.numpy())
        symbol = field.elements.shape_type.numpy()
        symbol = numpy.where(symbol == '?', 'asterisk', symbol)
        symbol = numpy.where(symbol == 'B', 'square', symbol)
        symbol = numpy.where(symbol == 'S', 'circle', symbol)
        symbol = symbol if symbol.shape else str(symbol)
        marker = graph_objects.scatter.Marker(size=marker_size, color=color, sizemode='diameter', symbol=symbol)
        fig.add_scatter(mode='markers', x=x, y=y, marker=marker, row=row, col=col)
        fig.update_xaxes(range=[lower_x, upper_x])
        fig.update_yaxes(range=[lower_y, upper_y])
        fig.update_layout(showlegend=False)
        subplot.xaxis.update(scaleanchor=f'y{subplot.yaxis.plotly_name[5:]}', scaleratio=1, constrain='domain')
        subplot.yaxis.update(constrain='domain')
    else:
        raise NotImplementedError(f"No figure recipe for {field}")


def real_values(field: SampledField):
    return field.values if field.values.dtype.kind != complex else abs(field.values)


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

