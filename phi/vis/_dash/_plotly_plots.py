import warnings
from typing import Tuple, Any, Dict, List, Callable, Union

import numpy
import numpy as np
from plotly import graph_objects, figure_factory
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS

from phi import math, field
from phi.field import SampledField, PointCloud, Grid, StaggeredGrid
from phi.geom import Sphere, BaseBox, Point, Box
from phi.geom._stack import GeometryStack
from phiml.math import Tensor, spatial, channel, non_channel
from phi.vis._dash.colormaps import COLORMAPS
from phi.vis._plot_util import smooth_uniform_curve, down_sample_curve
from phi.vis._vis_base import PlottingLibrary, Recipe


class PlotlyPlots(PlottingLibrary):

    def __init__(self):
        super().__init__('plotly', [graph_objects.Figure])

    def create_figure(self,
                      size: tuple,
                      rows: int,
                      cols: int,
                      subplots: Dict[Tuple[int, int], Box],
                      titles: Dict[Tuple[int, int], str],
                      log_dims: Tuple[str, ...]) -> Tuple[Any, Dict[Tuple[int, int], Any]]:
        titles = [titles.get((r, c), None) for r in range(rows) for c in range(cols)]
        specs = [[{'type': 'xy' if subplots.get((row, col), Box()).spatial_rank < 3 else 'surface'} for col in range(cols)] for row in range(rows)]
        fig = self.current_figure = make_subplots(rows=rows, cols=cols, subplot_titles=titles, specs=specs)
        for (row, col), bounds in subplots.items():
            subplot = fig.get_subplot(row + 1, col + 1)
            if bounds.spatial_rank == 1:
                subplot.xaxis.update(title=bounds.vector.item_names[0], range=_get_range(bounds, 0))
            elif bounds.spatial_rank == 2:
                subplot.xaxis.update(scaleanchor=f'y{subplot.yaxis.plotly_name[5:]}', scaleratio=1, constrain='domain', title=bounds.vector.item_names[0], range=_get_range(bounds, 0))
                subplot.yaxis.update(constrain='domain', title=bounds.vector.item_names[1], range=_get_range(bounds, 1))
            elif bounds.spatial_rank == 3:
                subplot.xaxis.update(title=bounds.vector.item_names[0], range=_get_range(bounds, 0))
                subplot.yaxis.update(title=bounds.vector.item_names[1], range=_get_range(bounds, 1))
                subplot.zaxis.update(title=bounds.vector.item_names[2], range=_get_range(bounds, 2))
        fig._phi_size = size
        return fig, {pos: (pos[0]+1, pos[1]+1) for pos in subplots.keys()}

    def animate(self, fig, frames: int, plot_frame_function: Callable, interval: float, repeat: bool):
        raise NotImplementedError()

    def finalize(self, figure):
        pass

    def close(self, figure):
        pass

    def show(self, figure: graph_objects.Figure):
        figure.show()

    def save(self, figure: graph_objects.Figure, path: str, dpi: float):
        width, height = figure._phi_size
        figure.layout.update(margin=dict(l=0, r=0, b=0, t=0))
        scale = dpi/90.
        figure.write_image(path, width=width * dpi / scale, height=height * dpi / scale, scale=scale)


class LinePlot(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return data.spatial_rank == 1 and data.is_grid

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        subplot = figure.get_subplot(row, col)
        x = data.points.vector[0].numpy().flatten()
        channels = data.values.shape.channel
        if channels.rank == 1 and channels.get_item_names(0) is not None:
            for i, name in enumerate(channels.get_item_names(0)):
                y = math.reshaped_native(real_values(data[{channels.name: i}]), [data.shape.spatial], to_numpy=True)
                figure.add_trace(graph_objects.Scatter(x=x, y=y, mode='lines+markers', name=name), row=row, col=col)
            figure.update_layout(showlegend=True)
        else:
            for ch_idx in channels.meshgrid():
                y = math.reshaped_native(real_values(data[ch_idx]), [data.shape.spatial], to_numpy=True)
                figure.add_trace(graph_objects.Scatter(x=x, y=y, mode='lines+markers', name='Multi-channel'), row=row, col=col)
            figure.update_layout(showlegend=False)
        if min_val is not None and max_val is not None:
            subplot.yaxis.update(range=(min_val - .02 * (max_val - min_val), max_val + .02 * (max_val - min_val)))


class Heatmap2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return data.spatial_rank == 2 and data.is_grid and 'vector' not in data.shape

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        dims = spatial(data)
        values = real_values(data).numpy(dims.reversed)
        x = data.points.vector[dims[0].name].dimension(dims[1].name)[0].numpy()
        y = data.points.vector[dims[1].name].dimension(dims[0].name)[0].numpy()
        min_val, max_val = numpy.nanmin(values), numpy.nanmax(values)
        min_val, max_val = min_val if numpy.isfinite(min_val) else 0, max_val if numpy.isfinite(max_val) else 0
        color_scale = get_div_map(min_val, max_val, equal_scale=True)
        # color_bar = graph_objects.heatmap.ColorBar(x=1.15)   , colorbar=color_bar
        figure.add_heatmap(row=row, col=col, x=x, y=y, z=values, zauto=False, zmin=min_val, zmax=max_val, colorscale=color_scale, showscale=show_color_bar)


class VectorField2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return data.spatial_rank == 2 and data.is_grid

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        if data.is_staggered:
            data = data.at_centers()
        row, col = subplot
        dims = data.bounds.vector.item_names
        vector = data.bounds.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        x, y = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
        u, v = math.reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.without(vector)])
        for ch in range(u.shape[0]):
            # quiver = figure_factory.create_quiver(x, y, data_x[ch], data_y[ch], scale=1.0)  # 7 points per arrow
            # fig.add_trace(quiver, row=row, col=col)
            u_ch = u[ch]
            v_ch = v[ch]
            # lines_y = numpy.stack([y, y + data_y_flat, [None] * len(x)], -1).flatten()  # 3 points per arrow
            # lines_x = numpy.stack([x, x + data_x_flat, [None] * len(x)], -1).flatten()
            lines_x = numpy.stack([x, x + u_ch, [None] * len(x)], -1).flatten()
            lines_y = numpy.stack([y, y + v_ch, [None] * len(x)], -1).flatten()  # 3 points per arrow
            name = extra_channels.get_item_names(0)[ch] if extra_channels.rank == 1 and extra_channels.get_item_names(0) is not None else None
            figure.add_scatter(x=lines_x, y=lines_y, mode='lines', row=row, col=col, name=name)
        if u.shape[0] == 1:
            figure.update_layout(showlegend=False)


class Heatmap3D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return data.spatial_rank == 3 and data.is_grid and data.shape.channel.volume == 1

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        dims = data.bounds.vector.item_names
        vector = data.bounds.shape['vector']
        values = real_values(data).numpy(dims)
        x, y, z = math.reshaped_numpy(data.points.vector[dims], [vector, *data.points.shape.spatial])
        min_val, max_val = numpy.nanmin(values), numpy.nanmax(values)
        min_val, max_val = min_val if numpy.isfinite(min_val) else 0, max_val if numpy.isfinite(max_val) else 0
        color_scale = get_div_map(min_val, max_val, equal_scale=True)
        figure.add_volume(x=x.flatten(), y=y.flatten(), z=z.flatten(), value=values.flatten(),
                          showscale=show_color_bar, colorscale=color_scale, cmin=min_val, cmax=max_val, cauto=False,
                          isomin=0.1, isomax=0.8,
                          opacity=0.1,  # needs to be small to see through all surfaces
                          surface_count=17,  # needs to be a large number for good volume rendering
                          row=row, col=col)
        figure.update_layout(uirevision=True)


class VectorField3D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return data.is_grid and data.spatial_rank == 3

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        dims = data.bounds.vector.item_names
        vector = data.bounds.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        if data.is_staggered:
            data = data.at_centers()
        x, y, z = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
        u, v, w = math.reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.non_channel])
        figure.add_cone(x=x.flatten(), y=y.flatten(), z=z.flatten(), u=u.flatten(), v=v.flatten(), w=w.flatten(),
                        colorscale='Blues',
                        sizemode="absolute", sizeref=1,
                        row=row, col=col)


class VectorCloud2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 2 and 'vector' in channel(data)

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        vector = data.bounds.shape['vector']
        x, y = math.reshaped_numpy(data.points, [vector, data.shape.without('vector')])
        u, v = math.reshaped_numpy(data.values, [vector, data.shape.without('vector')])
        quiver = figure_factory.create_quiver(x, y, u, v, scale=1.0).data[0]  # 7 points per arrow
        if (color != None).all:
            quiver.line.update(color=color.native())
        figure.add_trace(quiver, row=row, col=col)


class PointCloud2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 2

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        if isinstance(data.elements, GeometryStack):
            for idx in data.elements.geometries.shape[0].meshgrid():
                self.plot(data[idx], figure, subplot, space, min_val, max_val, show_color_bar, color[idx], alpha, err)
            return
        row, col = subplot
        subplot = figure.get_subplot(row, col)
        dims = data.bounds.vector.item_names
        vector = data.bounds.shape['vector']
        size = figure._phi_size
        yrange = subplot.yaxis.range
        if spatial(data):
            raise NotImplementedError("Plotly does not yet support plotting point clouds with spatial dimensions")
        for idx in non_channel(data.points).meshgrid(names=True):
            x, y = math.reshaped_numpy(data[idx].points.vector[dims], [vector, data.shape.non_channel])
            hex_color = color[idx].native()
            subplot_height = (subplot.yaxis.domain[1] - subplot.yaxis.domain[0]) * size[1] * 100
            if isinstance(data.elements, Sphere):
                symbol = 'circle'
                marker_size = data.elements.bounding_radius().numpy() * 1.9
            elif isinstance(data.elements, BaseBox):
                symbol = 'square'
                marker_size = math.mean(data.elements.bounding_half_extent(), 'vector').numpy() * 2
            elif isinstance(data.elements, Point):
                symbol = 'x'
                marker_size = 12 / (subplot_height / (yrange[1] - yrange[0]))
            else:
                symbol = 'asterisk'
                marker_size = data.elements.bounding_radius().numpy()
            marker_size *= subplot_height / (yrange[1] - yrange[0])
            marker = graph_objects.scatter.Marker(size=marker_size, color=hex_color, sizemode='diameter', symbol=symbol)
            figure.add_scatter(mode='markers', x=x, y=y, marker=marker, row=row, col=col)
        figure.update_layout(showlegend=False)


class PointCloud3D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 3

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        subplot = figure.get_subplot(row, col)
        dims = data.bounds.vector.item_names
        vector = data.bounds.shape['vector']
        size = figure._phi_size
        yrange = subplot.yaxis.range
        if data.points.shape.non_channel.rank > 1:
            data_list = field.unstack(data, data.points.shape.non_channel[0].name)
            for d in data_list:
                self.plot(d, figure, (row, col), space, min_val, max_val, show_color_bar, color)
        else:
            x, y, z = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
            color = color.native()
            domain_y = figure.layout[subplot.plotly_name].domain.y
            if isinstance(data.elements, Sphere):
                symbol = 'circle'
                marker_size = data.elements.bounding_radius().numpy() * 2
            elif isinstance(data.elements, BaseBox):
                symbol = 'square'
                marker_size = math.mean(data.elements.bounding_half_extent(), 'vector').numpy() * 1
            elif isinstance(data.elements, Point):
                symbol = 'x'
                marker_size = 4 / (size[1] * (domain_y[1] - domain_y[0]) / (yrange[1] - yrange[0]) * 0.5)
            else:
                symbol = 'asterisk'
                marker_size = data.elements.bounding_radius().numpy()
            marker_size *= size[1] * (domain_y[1] - domain_y[0]) / (yrange[1] - yrange[0]) * 0.5
            marker = graph_objects.scatter3d.Marker(size=marker_size, color=color, sizemode='diameter', symbol=symbol)
            figure.add_scatter3d(mode='markers', x=x, y=y, z=z, marker=marker, row=row, col=col)
        figure.update_layout(showlegend=False)


def _get_range(bounds: Box, index: int):
    lower = bounds.lower.vector[index].numpy()
    upper = bounds.upper.vector[index].numpy()
    return lower, upper


def real_values(field: SampledField):
    return field.values if field.values.dtype.kind != complex else abs(field.values)


def get_div_map(zmin, zmax, equal_scale=False, colormap: str = None):
    """
    Args:
      colormap(list or array, optional): colormap defined as list of [fraction_val, red_frac, green_frac, blue_frac] (Default value = None)
      zmin: 
      zmax: 
      equal_scale:  (Default value = False)
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


def plot_scalars(curves: Union[tuple, list], labels, subplots=True, log_scale='', smooth: int = 1):
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


def _graph(label: str, x: np.ndarray, y: np.ndarray, smooth: int, index: int, max_points=2000):
    color = DEFAULT_PLOTLY_COLORS[index % len(DEFAULT_PLOTLY_COLORS)]
    if len(x) > len(y):
        x = x[:len(y)]
    if len(y) > len(x):
        y = y[:len(x)]
    curves = split_curve(np.stack([x, y], -1))
    low_res = [down_sample_curve(c, max_points) for c in curves]
    x, y = join_curves(low_res).T
    if smooth <= 1:
        return [graph_objects.Scatter(x=x, y=y, name=label, line=graph_objects.scatter.Line(color=color))]
    else:  # smooth
        smooth_curves = [smooth_uniform_curve(c, smooth) for c in curves]
        low_res_smooth = [down_sample_curve(c, max_points) for c in smooth_curves]
        smooth_x, smooth_y = join_curves(low_res_smooth).T
        transparent_color = f"rgba{color[3:-1]}, 0.4)"
        return [
            graph_objects.Scatter(x=x, y=y, line=graph_objects.scatter.Line(color=transparent_color, width=1), showlegend=False),
            graph_objects.Scatter(x=smooth_x, y=smooth_y, name=label, line=graph_objects.scatter.Line(color=color, width=3), mode='lines')
        ]


def split_curve(curve: np.ndarray) -> List[np.ndarray]:
    x = curve[..., 0]
    backtracks = numpy.argwhere(x[1:] < x[:-1])[:, 0] + 1
    if len(backtracks) == 0:
        return [curve]
    cuts = [0] + list(backtracks) + [curve.shape[-2]]
    return [curve[s:e] for s, e in zip(cuts[:-1], cuts[1:])]


def join_curves(curves: List[np.ndarray]) -> np.ndarray:
    curves = [np.append(np.array(c, numpy.float), [[numpy.nan, numpy.nan]], -2) for c in curves[:-1]] + [curves[-1]]
    return np.concatenate(curves, -2)


PLOTLY = PlotlyPlots()
PLOTLY.recipes.extend([
            LinePlot(),
            Heatmap2D(),
            VectorField2D(),
            VectorField3D(),
            VectorCloud2D(),
            Heatmap3D(),
            PointCloud2D(),
            PointCloud3D(),
        ])