import warnings
from typing import Tuple, Any, Dict, List, Callable, Union

import numpy
import numpy as np
import plotly.graph_objs
from plotly.graph_objs import layout
from scipy.sparse import csr_matrix, coo_matrix

from phiml.math import reshaped_numpy, dual, instance
from plotly import graph_objects, figure_factory
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS
import plotly.io as pio

from phi import math, field
from phi.field import Field
from phi.geom import Sphere, BaseBox, Point, Box, SDF, SDFGrid
from phi.geom._geom_ops import GeometryStack
from phi.math import Tensor, spatial, channel, non_channel
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
                      log_dims: Tuple[str, ...],
                      plt_params: Dict[str, Any]) -> Tuple[Any, Dict[Tuple[int, int], Any]]:
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
        if size[0] is not None:
            fig.update_layout(width=size[0] * 70)
        if size[1] is not None:
            fig.update_layout(height=size[1] * 70)  # 70 approximately matches matplotlib but it's not consistent
        return fig, {pos: (pos[0]+1, pos[1]+1) for pos in subplots.keys()}

    def animate(self, fig, frames: int, plot_frame_function: Callable, interval: float, repeat: bool):
        raise NotImplementedError()

    def finalize(self, figure):
        pass

    def close(self, figure):
        pass

    def show(self, figure: graph_objects.Figure):
        figure.show()

    def save(self, figure: graph_objects.Figure, path: str, dpi: float, transparent: bool):
        width, height = figure._phi_size
        figure.layout.update(margin=dict(l=0, r=0, b=0, t=0))
        scale = dpi/90.
        if path.endswith('.html'):
            plotly.io.write_html(figure, path, auto_play=True, default_width="100%", default_height="100%")
        elif path.endswith('.json'):
            raise NotImplementedError("Call Plotly functions directly to save JSON.")
        else:
            figure.write_image(path, width=width * dpi / scale, height=height * dpi / scale, scale=scale)


class LinePlot(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return spatial(data).rank == 1 and data.is_grid

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
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

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 2 and data.is_grid and 'vector' not in data.shape

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
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

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 2 and data.is_grid

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        data = data.at_centers()
        row, col = subplot
        dims = data.elements.vector.item_names
        vector = data.elements.shape['vector']
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

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 3 and data.is_grid and data.shape.channel.volume == 1

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        dims = data.elements.vector.item_names
        vector = data.elements.shape['vector']
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


class VectorCloud3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 3 and 'vector' in data.shape

    def plot(self,
             data: Field,
             figure,
             subplot,
             space: Box,
             min_val: float,
             max_val: float,
             show_color_bar: bool,
             color: Tensor,
             alpha: Tensor,
             err: Tensor):
        row, col = subplot
        dims = data.elements.vector.item_names
        vector = data.elements.shape['vector']
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

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 2 and 'vector' in channel(data)

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        vector = data.elements.shape['vector']
        x, y = math.reshaped_numpy(data.points, [vector, data.shape.without('vector')])
        u, v = math.reshaped_numpy(data.values, [vector, data.shape.without('vector')])
        quiver = figure_factory.create_quiver(x, y, u, v, scale=1.0).data[0]  # 7 points per arrow
        if (color != None).all:
            quiver.line.update(color=color.native())
        figure.add_trace(quiver, row=row, col=col)


class PointCloud2D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 2

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        if isinstance(data.elements, GeometryStack):
            for idx in data.elements.geometries.shape[0].meshgrid():
                self.plot(data[idx], figure, subplot, space, min_val, max_val, show_color_bar, color[idx], alpha, err)
            return
        row, col = subplot
        subplot = figure.get_subplot(row, col)
        dims = data.elements.vector.item_names
        vector = data.elements.shape['vector']
        size = figure._phi_size
        yrange = subplot.yaxis.range
        if spatial(data):
            raise NotImplementedError("Plotly does not yet support plotting point clouds with spatial dimensions")
        for idx in non_channel(data.points).meshgrid(names=True):
            x, y = reshaped_numpy(data.points[idx].vector[dims], [vector, non_channel(data)])
            hex_color = color[idx].native()
            if hex_color is None:
                hex_color = pio.templates[pio.templates.default].layout.colorway[0]
            alphas = reshaped_numpy(alpha, [non_channel(data)])
            if isinstance(data.geometry, Sphere):
                hex_color = [hex_color] * non_channel(data).volume if isinstance(hex_color, str) else hex_color
                rad = reshaped_numpy(data.geometry.bounding_radius(), [data.shape.non_channel])
                for xi, yi, ri, ci, a in zip(x, y, rad, hex_color, alphas):
                    figure.add_shape(type="circle", xref="x", yref="y", x0=xi-ri, y0=yi-ri, x1=xi+ri, y1=yi+ri, fillcolor=ci, line_width=0)
            elif isinstance(data.geometry, BaseBox):
                hex_color = [hex_color] * non_channel(data).volume if isinstance(hex_color, str) else hex_color
                half_size = data.geometry.half_size
                min_len = space.size.sum
                half_size = math.where(math.is_finite(half_size), half_size, min_len)
                w2, h2 = reshaped_numpy(half_size, ['vector', data.shape.non_channel])
                if data.geometry.rotation_matrix is None:
                    lower_x = x - w2
                    lower_y = y - h2
                    upper_x = x + w2
                    upper_y = y + h2
                    for lxi, lyi, uxi, uyi, ci, a in zip(lower_x, lower_y, upper_x, upper_y, hex_color, alphas):
                        figure.add_shape(type="rect", xref="x", yref="y", x0=lxi, y0=lyi, x1=uxi, y1=uyi, fillcolor=ci, line_width=.5, line_color='#FFFFFF')
                else:
                    corners = data.geometry.corners
                    c4, c1, c3, c2 = reshaped_numpy(corners, [corners.shape.only(['~'+d for d in dims], reorder=True), non_channel(data), 'vector'])
                    for c1i, c2i, c3i, c4i, ci, a in zip(c1, c2, c3, c4, hex_color, alphas):
                        path = f"M{c1i[0]},{c1i[1]} L{c2i[0]},{c2i[1]} L{c3i[0]},{c3i[1]} L{c4i[0]},{c4i[1]} Z"
                        figure.add_shape(type="path", xref="x", yref="y", path=path, fillcolor=ci, line_width=.5, line_color='#FFFFFF')
            elif isinstance(data.geometry, SDFGrid):
                sdf = data.geometry.values.numpy(dims)
                x, y = data.geometry.points.numpy(('vector',) + dims)
                x = x[:, 0]
                y = y[0, :]
                dx, dy = data.geometry.dx[dims].numpy()
                sdf = np.where(sdf > (dx ** 2 + dy ** 2) ** .5, np.nan, sdf)
                colorscale = [[0, 'blue'], [.5, 'white'], [1, 'rgba(0.,0.,0.,0.)']]
                contour = go.Contour(
                    x=x, y=y, z=sdf.T,
                    contours=dict(start=-.001, end=.001, size=.002),
                    colorscale=colorscale, showscale=False,
                )
                figure.add_trace(contour)
            else:
                subplot_height = (subplot.yaxis.domain[1] - subplot.yaxis.domain[0]) * size[1] * 100 if size[1] is not None else None
                if isinstance(data.elements, Point):
                    symbol = None
                    marker_size = 12 / (subplot_height / (yrange[1] - yrange[0])) if subplot_height else None
                else:
                    symbol = 'asterisk'
                    marker_size = data.elements.bounding_radius().numpy()
                if subplot_height:
                    marker_size *= subplot_height / (yrange[1] - yrange[0])
                marker = graph_objects.scatter.Marker(size=marker_size, color=hex_color, sizemode='diameter', symbol=symbol)
                figure.add_scatter(mode='markers', x=x, y=y, marker=marker, row=row, col=col)
        figure.update_layout(showlegend=False)


class PointCloud3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 3

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        subplot = figure.get_subplot(row, col)
        dims = data.elements.vector.item_names
        vector = data.elements.shape['vector']
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
                symbol = None
                marker_size = 4 / (size[1] * (domain_y[1] - domain_y[0]) / (yrange[1] - yrange[0]) * 0.5)
            else:
                symbol = 'asterisk'
                marker_size = data.elements.bounding_radius().numpy()
            marker_size *= size[1] * (domain_y[1] - domain_y[0]) / (yrange[1] - yrange[0]) * 0.5
            marker = graph_objects.scatter3d.Marker(size=marker_size, color=color, sizemode='diameter', symbol=symbol)
            figure.add_scatter3d(mode='markers', x=x, y=y, z=z, marker=marker, row=row, col=col)
        figure.update_layout(showlegend=False)


class Graph3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_graph and data.spatial_rank == 3

    def plot(self,
             data: Field,
             figure: plotly.graph_objs.Figure,
             subplot,
             space: Box,
             min_val: float,
             max_val: float,
             show_color_bar: bool,
             color: Tensor,
             alpha: Tensor,
             err: Tensor):
        dims = space.vector.item_names
        row, col = subplot
        xyz = reshaped_numpy(data.graph.center.vector[dims], ['vector', instance])
        connectivity: coo_matrix = data.graph.connectivity.numpy().tocoo()
        x1, y1, z1 = xyz[:, connectivity.col]
        x2, y2, z2 = xyz[:, connectivity.row]
        x = np.stack([x1, x2, np.nan + x1], -1).flatten()
        y = np.stack([y1, y2, np.nan + y1], -1).flatten()
        z = np.stack([z1, z2, np.nan + z1], -1).flatten()
        figure.add_scatter3d(x=x, y=y, z=z, mode='lines', row=row, col=col)


class SurfaceMesh3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_mesh and data.spatial_rank == 3 and data.mesh.element_rank == 2 and not channel(data)

    def plot(self,
             data: Field,
             figure: plotly.graph_objs.Figure,
             subplot,
             space: Box,
             min_val: float,
             max_val: float,
             show_color_bar: bool,
             color: Tensor,
             alpha: Tensor,
             err: Tensor):
        dims = space.vector.item_names
        row, col = subplot
        x, y, z = reshaped_numpy(data.mesh.vertices.center.vector[dims], ['vector', instance])
        elements: csr_matrix = data.mesh.elements.numpy().tocsr()
        indices = elements.indices
        pointers = elements.indptr
        vertex_count = pointers[1:] - pointers[:-1]
        v1, v2, v3 = [], [], []
        # --- add triangles ---
        tris, = np.where(vertex_count == 3)
        tri_pointers = pointers[:-1][tris]
        v1.extend(indices[tri_pointers])
        v2.extend(indices[tri_pointers+1])
        v3.extend(indices[tri_pointers+2])
        # --- add two tris for each quad ---
        quads, = np.where(vertex_count == 4)
        quad_pointers = pointers[:-1][quads]
        v1.extend(indices[quad_pointers])
        v2.extend(indices[quad_pointers+1])
        v3.extend(indices[quad_pointers+2])
        v1.extend(indices[quad_pointers+1])
        v2.extend(indices[quad_pointers+2])
        v3.extend(indices[quad_pointers+3])
        # --- polygons with > 4 vertices ---
        if np.any(vertex_count > 4):
            warnings.warn("Only tris and quads are currently supported with Plotly mesh render", RuntimeWarning)
        # --- plot mesh ---
        cbar = None if not channel(data) or not channel(data).item_names[0] else channel(data).item_names[0][0]
        if math.is_nan(data.values).all:
            mesh = go.Mesh3d(x=x, y=y, z=z, i=v1, j=v2, k=v3)
        else:
            values = reshaped_numpy(data.values, [instance(data.mesh)])
            mesh = go.Mesh3d(x=x, y=y, z=z, i=v1, j=v2, k=v3, colorscale='viridis', colorbar_title=cbar, intensity=values, intensitymode='cell')
        figure.add_trace(mesh, row=row, col=col)


class SDF3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return isinstance(data.geometry, SDF) and data.spatial_rank == 3

    def plot(self,
             data: Field,
             figure: go.Figure,
             subplot,
             space: Box,
             min_val: float,
             max_val: float,
             show_color_bar: bool,
             color: Tensor,
             alpha: Tensor,
             err: Tensor):
        # surf_mesh = mesh_from_sdf(data.geometry, remove_duplicates=True, backend=math.NUMPY)
        # mesh_data = Field(surf_mesh, math.NAN, 0)
        # SurfaceMesh3D().plot(mesh_data, figure, subplot, space, min_val, max_val, show_color_bar, color, alpha, err)
        from sdf.mesh import generate  # using https://github.com/fogleman/sdf
        mesh = np.stack(generate(data.geometry, workers=1, batch_size=1024*1024))
        # --- remove duplicate vertices ---
        vert, idx, inv, c = np.unique(mesh, axis=0, return_counts=True, return_index=True, return_inverse=True)
        i, j, k = inv.reshape((-1, 3)).T
        mesh = go.Mesh3d(x=vert[:, 0], y=vert[:, 1], z=vert[:, 2], i=i, j=j, k=k)
        figure.add_trace(mesh)
        # fig = go.Figure(go.Mesh3d(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2], i=i, j=i+1, k=i+2))
        # fig.show()



def _get_range(bounds: Box, index: int):
    lower = bounds.lower.vector[index].numpy()
    upper = bounds.upper.vector[index].numpy()
    return lower, upper


def real_values(field: Field):
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
    # --- 2D ---
    Heatmap2D(),
    VectorField2D(),
    VectorCloud2D(),
    PointCloud2D(),
    # --- 3D ---
    Heatmap3D(),
    SurfaceMesh3D(),
    SDF3D(),
    Graph3D(),
    VectorCloud3D(),
    PointCloud3D(),
])