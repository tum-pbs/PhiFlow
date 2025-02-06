import os
import subprocess
import tempfile
import warnings
import webbrowser
from numbers import Number
from typing import Tuple, Any, Dict, List, Callable, Union, Optional

import numpy
import numpy as np
import plotly.graph_objs

from phiml.math._sparse import CompactSparseTensor
from scipy.sparse import csr_matrix, coo_matrix

from plotly import graph_objects, figure_factory
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS

from phiml.math import reshaped_numpy, dual, instance, non_dual, merge_shapes, pack_dims, dsum, close, equal, NAN
from phi import math, geom
from phi.field import Field
from phi.geom import Sphere, BaseBox, Point, Box, SDF, SDFGrid, Cylinder, Mesh
from phi.geom._geom_ops import GeometryStack
from phi.math import Tensor, spatial, channel, non_channel
from phi.vis._dash.colormaps import COLORMAPS
from phi.vis._plot_util import smooth_uniform_curve, down_sample_curve
from phi.vis._vis_base import PlottingLibrary, Recipe, is_jupyter, display_name
from phiml.math._tensors import Layout


class PlotlyPlots(PlottingLibrary):

    def __init__(self):
        super().__init__('plotly', [graph_objects.Figure])

    def create_figure(self,
                      size: tuple,
                      rows: int,
                      cols: int,
                      subplots: Dict[Tuple[int, int], Box],
                      log_dims: Tuple[str, ...],
                      plt_params: Dict[str, Any]) -> Tuple[Any, Dict[Tuple[int, int], Any]]:
        specs = [[{'type': 'xy' if subplots.get((row, col), Box()).spatial_rank < 3 else 'surface'} for col in range(cols)] for row in range(rows)]
        fig = self.current_figure = make_subplots(rows=rows, cols=cols, specs=specs)
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
                subplot.aspectmode = 'manual'
                x_range = _get_range(bounds, 0)
                y_range = _get_range(bounds, 1)
                z_range = _get_range(bounds, 2)
                n = float(math.length(bounds.size).max) * .5
                subplot.aspectratio = dict(x=(x_range[1]-x_range[0])/n, y=(y_range[1]-y_range[0])/n, z=(z_range[1]-z_range[0])/n)
        fig._phi_size = size
        if size[0] is not None:
            fig.update_layout(width=size[0] * 70)
        if size[1] is not None:
            fig.update_layout(height=size[1] * 70)  # 70 approximately matches matplotlib but it's not consistent
        return fig, {pos: (pos[0]+1, pos[1]+1) for pos in subplots.keys()}

    def set_title(self, title, figure: go.Figure, subplot):
        if subplot is not None:
            subplot = figure.get_subplot(*subplot)
            if hasattr(subplot, 'domain'):
                domain = subplot.domain.x, subplot.domain.y
            else:
                domain = [subplot.xaxis.domain, subplot.yaxis.domain]
            annotation = _build_subplot_title_annotations([title], domain)
            figure.layout.annotations += tuple(annotation)
        else:
            figure.update_layout(title_text=title)

    def animate(self, fig, frame_count: int, plot_frame_function: Callable, interval: float, repeat: bool, interactive: bool, time_axis: Optional[str]):
        figures = []
        for frame in range(frame_count):
            frame_fig = go.Figure(fig)
            frame_fig._phi_size = fig._phi_size
            plot_frame_function(frame_fig, frame)
            figures.append(frame_fig)
        frames = [go.Frame(data=fig.data, layout=fig.layout, name=f'frame{i}') for i, fig in enumerate(figures)]
        anim = go.Figure(data=figures[0].data, layout=figures[0].layout, frames=frames)
        anim._phi_size = fig._phi_size
        if interactive:
            names = [f.layout.title.text if f.layout.title.text else f'{i}' for i, f in enumerate(figures)]
            anim.update_layout(
                updatemenus=[{
                    'buttons': [
                        {
                            'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                            'label': '⏵',
                            'method': 'animate'
                        },
                        {
                            'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                            'label': '⏸',
                            'method': 'animate'
                        }
                    ],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 87},
                    'showactive': False,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }],
                sliders=[{
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'font': {'size': 20},
                        'prefix': display_name(time_axis) + " ",
                        'visible': True,
                        'xanchor': 'right'
                    },
                    'transition': {'duration': interval, 'easing': 'cubic-in-out'},
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': [{
                        'args': [[f'frame{i}'], {'frame': {'duration': interval, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': interval}}],
                        'label': names[i],
                        'method': 'animate'
                    } for i in range(frame_count)]
                }]
            )
        return anim

    def finalize(self, figure):
        pass

    def close(self, figure):
        pass

    def show(self, figure: graph_objects.Figure):
        if is_jupyter():
            plotly.io.renderers.default = 'notebook'
        else:
            plotly.io.renderers.default = 'browser'
        try:
            figure.show()
        except webbrowser.Error as err:
            warnings.warn(f"{err}", RuntimeWarning)

    def save(self, figure: graph_objects.Figure, path: str, dpi: float, transparent: bool):
        width, height = figure._phi_size
        figure.layout.update(margin=dict(l=0, r=0, b=0, t=0))
        scale = dpi/90.
        width = None if width is None else width * dpi / scale
        height = None if height is None else height * dpi / scale
        if path.endswith('.html'):
            plotly.io.write_html(figure, path, auto_play=True, default_width="100%", default_height="100%")
        elif path.endswith('.json'):
            raise NotImplementedError("Call Plotly functions directly to save JSON.")
        elif path.endswith('.mp4'):
            config = {
                'displayModeBar': False,  # Hides the modebar
                'displaylogo': False,  # Hides the Plotly logo
                'scrollZoom': False,  # Disables scroll-to-zoom
                'showAxisDragHandles': False,  # Hides axis drag handles
                'staticPlot': True,  # Makes the plot static (no interaction)
            }
            img_dir = tempfile.mkdtemp()
            images = []
            for i, frame in enumerate(figure.frames):
                layout = frame.layout
                layout.update(sliders=[], updatemenus=[])
                frame_fig = go.Figure(data=frame.data, layout=layout)
                img_path = os.path.join(img_dir, f'{i:04d}.png')
                print(f"Writing image to {img_path}... (width={width}, height={height}, scale={scale})")
                frame_fig.write_image(img_path, width=width, height=height, scale=scale)  # requires kaleido==0.1.0.post1, see https://community.plotly.com/t/static-image-export-hangs-using-kaleido/61519/3
                images.append(img_path)
                # image = PIL.Image.open(io.BytesIO(figure.to_image(format="png")))
            frame_rate = 1 / .2
            print("Writing video...")
            command = [
                'ffmpeg',
                '-y',  # override
                '-framerate', str(frame_rate),
                '-i', os.path.join(img_dir, f'%04d.png'),  # Assuming image files are named like 001.png, 002.png, etc.
                # '-c:v', 'libx264',
                # '-pix_fmt', 'yuv420p',
                path
            ]
            subprocess.run(command, check=True)
        else:
            figure.write_image(path, width=width, height=height, scale=scale)


class LinePlot(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return spatial(data).rank == 1 and data.is_grid

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        subplot = figure.get_subplot(row, col)
        x = data.points.vector[0].numpy().flatten()
        channels = data.values.shape.channel
        if channels.rank == 1 and channels.item_names[0] is not None:
            for i, name in enumerate(channels.item_names[0]):
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
            name = extra_channels.item_names[0][ch] if extra_channels.rank == 1 and extra_channels.item_names[0] is not None else None
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
                          showscale=show_color_bar, colorscale=color_scale, cmin=min_val, cmax=max_val,
                          opacity=0.2 * float(alpha),  # needs to be small to see through all surfaces
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
        if color == 'cmap':
            colorscale = 'Blues'
        else:
            hex_color = plotly_color(color)
            colorscale = [[0, hex_color], [1, hex_color]]
        if data.is_staggered:
            data = data.at_centers()
        x, y, z = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
        u, v, w = math.reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.non_channel])
        figure.add_cone(x=x.flatten(), y=y.flatten(), z=z.flatten(), u=u.flatten(), v=v.flatten(), w=w.flatten(),
                        colorscale=colorscale,
                        sizemode='raw', anchor='tail',
                        row=row, col=col, opacity=float(alpha))


class VectorCloud2D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 2 and 'vector' in channel(data)

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        vector = data.elements.shape['vector']
        x, y = math.reshaped_numpy(data.points, [vector, data.shape.without('vector')])
        u, v = math.reshaped_numpy(data.values, [vector, data.shape.without('vector')])
        quiver = figure_factory.create_quiver(x, y, u, v, scale=1.0).data[0]  # 7 points per arrow
        if color != 'cmap':
            quiver.line.update(color=plotly_color(color))
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
            if color[idx] == 'cmap':
                hex_color = plotly_color(0)  # ToDo add color bar
            else:
                hex_color = plotly_color(color[idx])
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


class Object3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        if not data.is_point_cloud or data.spatial_rank != 3:
            return False
        if isinstance(data.geometry, Sphere):
            v_count = self._sphere_vertex_count(data.geometry.radius, space)
            face_count = (v_count + 1) * (v_count * 2) / 2  # half as many tris as vertices
        elif isinstance(data.geometry, BaseBox):
            face_count = 12
        elif isinstance(data.geometry, Cylinder):
            v_count = self._sphere_vertex_count(data.geometry.radius, space)
            face_count = 2 + v_count
        else:
            return False
        face_count *= non_dual(data.geometry).without('vector').volume
        return face_count < 10_000

    def plot(self, data: Field, figure: go.Figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        dims = data.geometry.vector.item_names
        def plot_one_material(data, color, alpha: float):
            if color == 'cmap':
                color = plotly_color(0)  # ToDo cmap
            else:
                color = plotly_color(color)
            alpha = float(alpha)
            count = instance(data.geometry).volume
            if isinstance(data.geometry, Sphere):
                vertex_count = self._sphere_vertex_count(data.geometry.radius, space)
                cx, cy, cz = reshaped_numpy(data.points[dims], ['vector', instance, (), ()])
                rad = reshaped_numpy(data.geometry.radius, [instance, (), ()])
                d = np.pi / vertex_count
                theta, phi = np.mgrid[0:np.pi+d:d, 0:2*np.pi:d]
                # --- to cartesian ---
                x = np.sin(theta) * np.cos(phi) * rad + cx
                y = np.sin(theta) * np.sin(phi) * rad + cy
                z = np.cos(theta) * rad + cz
                for inst in range(count):
                    xyz = np.vstack([x[inst].ravel(), y[inst].ravel(), z[inst].ravel()])
                    figure.add_trace(go.Mesh3d(x=xyz[0], y=xyz[1], z=xyz[2], flatshading=False, alphahull=0, color=color, opacity=alpha), row=row, col=col)
            elif isinstance(data.geometry, Cylinder):
                vertex_count = self._sphere_vertex_count(data.geometry.radius, space)
                x, y, z = data.geometry.vertex_rings(dual(vertices=vertex_count)).numpy(['vector', instance, dual])
                for inst in range(count):
                    figure.add_trace(go.Mesh3d(x=x[inst], y=y[inst], z=z[inst], flatshading=False, alphahull=0, color=color, opacity=alpha), row=row, col=col)
            elif isinstance(data.geometry, BaseBox):
                cx, cy, cz = reshaped_numpy(data.geometry.corners, ['vector', instance, *['~' + d for d in dims]])
                x = cx.flatten()
                y = cy.flatten()
                z = cz.flatten()
                v1 = [0, 1, 4, 5, 4, 6, 5, 7, 0, 1, 2, 6]
                v2 = [1, 3, 6, 6, 0, 0, 7, 3, 4, 4, 3, 3]
                v3 = [2, 2, 5, 7, 6, 2, 1, 1, 1, 5, 6, 7]
                v1 = (v1 + np.arange(count)[:, None] * 8).flatten()
                v2 = (v2 + np.arange(count)[:, None] * 8).flatten()
                v3 = (v3 + np.arange(count)[:, None] * 8).flatten()
                figure.add_trace(go.Mesh3d(x=x, y=y, z=z, i=v1, j=v2, k=v3, flatshading=False, color=color, opacity=alpha), row=row, col=col)
        math.map(plot_one_material, data, color, alpha, dims=merge_shapes(color, alpha), unwrap_scalars=True)

    def _sphere_vertex_count(self, radius: Tensor, space: Box):
        with math.NUMPY:
            radius = math.convert(radius)
            space = math.convert(space)
        size_in_fig = radius.max / space.size.max
        def individual_vertex_count(size_in_fig):
            if ~np.isfinite(size_in_fig):
                return 0
            return np.clip(int(abs(size_in_fig) ** .5 * 50), 4, 64)
        return math.map(individual_vertex_count, size_in_fig)


class Scatter3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 3

    def plot(self, data: Field, figure: go.Figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        row, col = subplot
        subplot = figure.get_subplot(row, col)
        dims = data.elements.vector.item_names
        vector = data.elements.shape['vector']
        size = (figure._phi_size[0] or 10, figure._phi_size[1] or 6)
        yrange = subplot.yaxis.range
        for idx in (channel(data.geometry) - 'vector').meshgrid():
            if color == 'cmap':
                color_i = data[idx].values.numpy([math.shape]).astype(np.float32)
            else:
                color_i = plotly_color(color[idx])
            if spatial(data.geometry):
                for sdim in spatial(data.geometry):
                    xyz = math.reshaped_numpy(data[idx].points.vector[dims], [vector, ..., sdim])
                    xyz_padded = [[i.tolist() + [None] for i in c] for c in xyz]
                    x, y, z = [sum(c, []) for c in xyz_padded]
                    mode = 'markers+lines' if data.shape.non_channel.volume <= 100 else 'lines'
                    figure.add_scatter3d(mode=mode, x=x, y=y, z=z, row=row, col=col, line=dict(color=color_i, width=2), opacity=float(alpha))
                continue
            # if data.points.shape.non_channel.rank > 1:
            #     data_list = field.unstack(data, data.points.shape.non_channel[0].name)
            #     for d in data_list:
            #         self.plot(d, figure, (row, col), space, min_val, max_val, show_color_bar, color_i, alpha, err)
            #     return
            domain_y = figure.layout[subplot.plotly_name].domain.y
            x, y, z = math.reshaped_numpy(data[idx].points.vector[dims], [vector, data.shape.non_channel])
            if isinstance(data.geometry, Sphere):
                symbol = 'circle'
                marker_size = data.geometry[idx].bounding_radius().numpy() * 2
            elif isinstance(data.geometry, BaseBox):
                symbol = 'square'
                marker_size = math.mean(data.geometry[idx].bounding_half_extent(), 'vector').numpy() * 1
            elif isinstance(data.geometry, Point):
                symbol = None
                marker_size = 4 / (size[1] * (domain_y[1] - domain_y[0]) / (yrange[1] - yrange[0]) * 0.5)
            else:
                symbol = 'diamond-open'
                marker_size = 20
            marker_size *= size[1] * (domain_y[1] - domain_y[0]) / (yrange[1] - yrange[0]) * 0.5
            marker = graph_objects.scatter3d.Marker(size=marker_size, color=color_i, colorscale='Viridis', sizemode='diameter', symbol=symbol)
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
        batch_dims = data.mesh.elements.shape.only(data.mesh.vertices.shape)
        if batch_dims.only(color.shape):
            # must plot multiple meshes, as color cannot be specified per-face
            for bi in batch_dims.only(color.shape).meshgrid():
                self.plot(Field(data.geometry[bi], NAN), figure, subplot, space, min_val, max_val, show_color_bar, color[bi], alpha[bi], err[bi])
            return
        v_offset = [0]
        e_offset = [0]
        v1, v2, v3 = [], [], []
        for bi in batch_dims.meshgrid():
            mesh = data.mesh[bi]
            if isinstance(mesh.elements, CompactSparseTensor):
                polygons = mesh.elements._indices
                math.assert_close(1, mesh.elements._values)
                if dual(polygons).size == 3:  # triangles
                    t1, t2, t3 = polygons.numpy([dual, instance]) + v_offset[-1]
                    v1.extend(t1)
                    v2.extend(t2)
                    v3.extend(t3)
                else:
                    q1, q2, q3, q4 = polygons.numpy([dual, instance]) + v_offset[-1]
                    v1.extend(np.concatenate([q1, q1]))
                    v2.extend(np.concatenate([q2, q3]))
                    v3.extend(np.concatenate([q3, q4]))
            else:
                elements: csr_matrix = mesh.elements.numpy().tocsr()
                indices = elements.indices + v_offset[-1]
                pointers = elements.indptr
                vertex_count = pointers[1:] - pointers[:-1]
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
                v1.extend(indices[quad_pointers])
                v2.extend(indices[quad_pointers+2])
                v3.extend(indices[quad_pointers+3])
                # --- polygons with > 4 vertices ---
                if np.any(vertex_count > 4):
                    warnings.warn("Only tris and quads are currently supported with Plotly mesh render", RuntimeWarning)
            v_offset.append(v_offset[-1] + instance(mesh.vertices).volume)
            e_offset.append(len(v1))
        # --- plot mesh ---
        cbar_title = None if not channel(data) or not channel(data).item_names[0] else channel(data).item_names[0][0]
        if math.is_nan(data.values).all:
            mesh = go.Mesh3d(x=x, y=y, z=z, i=v1, j=v2, k=v3, flatshading=False, opacity=float(alpha), color=plotly_color(color))
        elif data.sampled_at == 'center':
            if any(b in data.values.shape for b in batch_dims):
                values = []
                for bi in batch_dims.meshgrid():
                    vertex_count = dsum(data.mesh[bi].elements).numpy()
                    repeat = np.where(vertex_count == 4, 2, 1)
                    bi_values = reshaped_numpy(data.values[bi], [instance(data.mesh)-batch_dims])
                    bi_values = np.repeat(bi_values, repeat)
                    values.extend(bi_values)
            else:
                values = reshaped_numpy(data.values, [[batch_dims + instance(data.mesh)]])
            mesh = go.Mesh3d(x=x, y=y, z=z, i=v1, j=v2, k=v3, colorscale='viridis', colorbar_title=cbar_title, intensity=values, intensitymode='cell', flatshading=True, opacity=float(alpha))
        elif data.sampled_at == 'vertex':
            values = reshaped_numpy(data.values, [instance(data.mesh.vertices)])
            mesh = go.Mesh3d(x=x, y=y, z=z, i=v1, j=v2, k=v3, colorscale='viridis', colorbar_title=cbar_title, intensity=values, intensitymode='vertex', flatshading=True, opacity=float(alpha))
        elif data.sampled_at == '~vertex':
            values = reshaped_numpy(data.values, [dual(data.mesh.elements)])
            mesh = go.Mesh3d(x=x, y=y, z=z, i=v1, j=v2, k=v3, colorscale='viridis', colorbar_title=cbar_title, intensity=values, intensitymode='vertex', flatshading=True, opacity=float(alpha))
        else:
            warnings.warn(f"No recipe for mesh sampled at {data.sampled_at}")
            return
        figure.add_trace(mesh, row=row, col=col)


class SDF3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return isinstance(data.geometry, (SDF, SDFGrid)) and data.spatial_rank == 3

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
        def plot_single_material(data: Field, color, alpha: float):
            with math.NUMPY:
                surf_mesh = geom.surface_mesh(data.geometry)
            mesh_data = Field(surf_mesh, math.NAN, 0)
            SurfaceMesh3D().plot(mesh_data, figure, subplot, space, min_val, max_val, show_color_bar, color, alpha, err)
        math.map(plot_single_material, data, color, alpha, dims=channel(data.geometry) - 'vector', unwrap_scalars=False)


def _get_range(bounds: Box, index: int):
    lower = float(bounds.lower.vector[index])
    upper = float(bounds.upper.vector[index])
    return lower, upper


def real_values(field: Field):
    return field.values if field.values.dtype.kind != complex else abs(field.values)


def plotly_color(col: Union[int, str, np.ndarray, Tensor]):
    if isinstance(col, Tensor) and col.dtype.kind == object:
        col = col.native()
    elif isinstance(col, Tensor):
        col = col.numpy()
    if isinstance(col, np.ndarray):
        col = col.item()
    if isinstance(col, int):
        return DEFAULT_PLOTLY_COLORS[col % len(DEFAULT_PLOTLY_COLORS)]
    if isinstance(col, str) and (col.startswith('#') or col.startswith('rgb(') or col.startswith('rgba(')):
        return col
    elif isinstance(col, str):
        return col  # color name, e.g. 'blue'
    elif isinstance(col, Number) and int(col) == col:
        return DEFAULT_PLOTLY_COLORS[int(col) % len(DEFAULT_PLOTLY_COLORS)]
    raise NotImplementedError(type(col), col)
    # if isinstance(col, (tuple, list)):
    #     col = np.asarray(col)
    #     if col.dtype.kind == 'i':
    #         col = col / 255.
    #     return col

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
        below = offset_positions[offset_positions < 0]
        color1 = cm_arr[numpy.argmax(below)] if below.size > 0 else cm_arr[0]  # largest value smaller than control
        above = offset_positions[offset_positions > 0]
        color2 = cm_arr[numpy.argmin(above)] if above.size > 0 else cm_arr[-1]  # smallest value larger than control
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


def _build_subplot_title_annotations(subplot_titles, list_of_domains, title_edge="top", offset=0):  # copied from plotly for future compatibility
    # If shared_axes is False (default) use list_of_domains
    # This is used for insets and irregular layouts
    # if not shared_xaxes and not shared_yaxes:
    x_dom = list_of_domains[::2]
    y_dom = list_of_domains[1::2]
    subtitle_pos_x = []
    subtitle_pos_y = []

    if title_edge == "top":
        text_angle = 0
        xanchor = "center"
        yanchor = "bottom"

        for x_domains in x_dom:
            subtitle_pos_x.append(sum(x_domains) / 2.0)
        for y_domains in y_dom:
            subtitle_pos_y.append(y_domains[1])

        yshift = offset
        xshift = 0
    elif title_edge == "bottom":
        text_angle = 0
        xanchor = "center"
        yanchor = "top"

        for x_domains in x_dom:
            subtitle_pos_x.append(sum(x_domains) / 2.0)
        for y_domains in y_dom:
            subtitle_pos_y.append(y_domains[0])

        yshift = -offset
        xshift = 0
    elif title_edge == "right":
        text_angle = 90
        xanchor = "left"
        yanchor = "middle"

        for x_domains in x_dom:
            subtitle_pos_x.append(x_domains[1])
        for y_domains in y_dom:
            subtitle_pos_y.append(sum(y_domains) / 2.0)

        yshift = 0
        xshift = offset
    elif title_edge == "left":
        text_angle = -90
        xanchor = "right"
        yanchor = "middle"

        for x_domains in x_dom:
            subtitle_pos_x.append(x_domains[0])
        for y_domains in y_dom:
            subtitle_pos_y.append(sum(y_domains) / 2.0)

        yshift = 0
        xshift = -offset
    else:
        raise ValueError("Invalid annotation edge '{edge}'".format(edge=title_edge))

    plot_titles = []
    for index in range(len(subplot_titles)):
        if not subplot_titles[index] or index >= len(subtitle_pos_y):
            pass
        else:
            annot = {
                "y": subtitle_pos_y[index],
                "xref": "paper",
                "x": subtitle_pos_x[index],
                "yref": "paper",
                "text": subplot_titles[index],
                "showarrow": False,
                "font": dict(size=16),
                "xanchor": xanchor,
                "yanchor": yanchor,
            }

            if xshift != 0:
                annot["xshift"] = xshift

            if yshift != 0:
                annot["yshift"] = yshift

            if text_angle != 0:
                annot["textangle"] = text_angle

            plot_titles.append(annot)
    return plot_titles


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
    Object3D(),
    Scatter3D(),
])