import sys
import warnings
from typing import Callable, Tuple, Any, Dict, Union, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import rc
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from phi import math
from phi.field import StaggeredGrid, Field, CenteredGrid
from phi.geom import Sphere, BaseBox, Point, Box, Mesh, Graph, SDFGrid, SDF, UniformGrid, rotate, rotation_angles
from phi.geom._heightmap import Heightmap
from phi.geom._geom_ops import GeometryStack
from phi.geom._embed import _EmbeddedGeometry
from phi.math import Tensor, channel, spatial, instance, non_channel, Shape, reshaped_numpy, shape
from phi.vis._vis_base import display_name, PlottingLibrary, Recipe, index_label, only_stored_elements, to_field
from phiml.math import wrap

colormaps = matplotlib.colormaps if hasattr(matplotlib.colormaps, 'get_cmap') else matplotlib.cm


class MatplotlibPlots(PlottingLibrary):

    def __init__(self):
        super().__init__('matplotlib', [plt.Figure, animation.Animation])

    def create_figure(self,
                      size: tuple,
                      rows: int,
                      cols: int,
                      spaces: Dict[Tuple[int, int], Box],
                      log_dims: Tuple[str, ...],
                      plt_params: Dict[str, Any]) -> Tuple[Any, Dict[Tuple[int, int], Any]]:
        size = (size[0] or 12, size[1] or 5)
        figure, axes = plt.subplots(rows, cols, figsize=size)
        self.current_figure = figure
        axes = np.reshape(axes, (rows, cols))
        axes_by_pos = {}
        subplot_aspect = (size[0] / cols) / (size[1] / rows)  # x / y
        for row in range(rows):
            for col in range(cols):
                axis = axes[row, col]
                if (row, col) not in spaces:
                    axis.remove()
                else:
                    bounds = spaces[(row, col)]
                    if bounds.spatial_rank == 2:
                        x, y = bounds.vector.item_names
                        axis.set_xlabel(display_name(x))
                        axis.set_ylabel(display_name(y))
                        # --- Log axes ---
                        x_log = bounds.vector.item_names[0] in log_dims
                        y_log = bounds.vector.item_names[1] in log_dims
                        if x_log:
                            axis.set_xscale('log')
                        if y_log:
                            axis.set_yscale('log')
                        # --- limits ---
                        x_range, y_range = [_get_range(bounds, i) for i in (0, 1)]
                        if None not in x_range:
                            if x_log and x_range[0] <= 0:
                                x_range = (1e-3 * x_range[1], x_range[1])
                            axis.set_xlim(x_range)
                        if None not in y_range:
                            if y_log and y_range[0] <= 0:
                                y_range = (1e-3 * y_range[1], y_range[1])
                            axis.set_ylim(y_range)
                        # --- Equal aspect ---
                        max_aspect = 4 if all([n in ['x', 'y', 'z'] for n in bounds.vector.item_names]) else 1.5
                        if None not in x_range and None not in y_range and '_' not in bounds.vector.item_names:
                            x_size, y_size = x_range[1] - x_range[0], y_range[1] - y_range[0]
                            if not x_log and not y_log and x_size > 0 and y_size > 0 and max(x_size/y_size/subplot_aspect, y_size/x_size*subplot_aspect) < max_aspect:
                                axis.set_aspect('equal', adjustable='box')
                        # --- Remove labels if axes shared ---
                        for left_col in range(col):
                            if (row, left_col) in spaces and y in spaces[(row, left_col)].vector.item_names and spaces[(row, left_col)].vector[y] == bounds.vector[y] and math.is_finite(bounds.vector[y].lower).all:
                                axis.set_ylabel("")
                                axis.tick_params(labelleft=False)
                                axis.yaxis.set_minor_formatter(NullFormatter())  # sometimes required for log axis
                        for below_row in range(row + 1, rows + 1):
                            if (below_row, col) in spaces and x in spaces[(below_row, col)].vector.item_names and spaces[(below_row, col)].vector[x] == bounds.vector[x] and math.is_finite(bounds.vector[y].lower).all:
                                axis.set_xlabel("")
                                axis.tick_params(labelbottom=False)
                                axis.xaxis.set_minor_formatter(NullFormatter())  # sometimes required for log axis
                    elif bounds.spatial_rank == 3:
                        axis.remove()
                        auto_order = True
                        if 'z-order' in plt_params:
                            assert plt_params['z-order'] in ['dynamic', 'as-provided']
                            auto_order = plt_params['z-order'] == 'dynamic'
                        axis = axes[row, col] = figure.add_subplot(rows, cols, cols*row + col + 1, projection='3d', computed_zorder=auto_order)
                        if not auto_order:
                            axis._phi_z_order_index = 0
                        # --- limits ---
                        axis.set_xlabel(display_name(bounds.vector.item_names[0]))
                        axis.set_ylabel(display_name(bounds.vector.item_names[1]))
                        axis.set_zlabel(display_name(bounds.vector.item_names[2]))
                        axis.set_xlim(_get_range(bounds, 0))
                        axis.set_ylim(_get_range(bounds, 1))
                        axis.set_zlim(_get_range(bounds, 2))
                        # --- Equal aspect ---
                        if hasattr(axis, 'set_box_aspect'):
                            aspect3d = list(math.max(bounds.size, bounds.shape.without('vector')).numpy())
                            axis.set_box_aspect(aspect3d)
                        # --- Log axes ---
                        if bounds.vector.item_names[0] in log_dims:
                            warnings.warn("Only z axis can be log scaled in 3D plot with Matplotlib. Please reorder the dimensions.", RuntimeWarning)
                            # subplot.set_xscale('log')
                        if bounds.vector.item_names[1] in log_dims:
                            warnings.warn("Only z axis can be log scaled in 3D plot with Matplotlib. Please reorder the dimensions.", RuntimeWarning)
                            # subplot.set_yscale('log')
                        if bounds.vector.item_names[2] in log_dims:
                            axis.set_zscale('log')
                    axes_by_pos[(row, col)] = axes[row, col]
        try:
            figure.tight_layout()
        except ValueError as err:
            warnings.warn(f"tight_layout could not be applied: {err}")
        return figure, axes_by_pos

    def set_title(self, title: str, figure, subplot: Optional):
        if subplot is None:
            pass
        else:
            subplot.set_title(title)

    def animate(self, fig: plt.Figure, frame_count: int, plot_frame_function: Callable, interval: float, repeat: bool, interactive: bool, time_axis: Optional[str]):
        if 'ipykernel' in sys.modules:
            rc('animation', html='html5')

        base_axes = tuple(fig.axes)
        positions = {a: (a.get_subplotspec().get_position(a.figure).p0, a.get_subplotspec().get_position(a.figure).p1) for a in base_axes}
        # titles = {a: a.get_title() for a in base_axes}
        specs = {a: a.get_subplotspec() for a in base_axes}

        def clear_and_plot(frame: int):
            axes = tuple(fig.axes)
            for axis in axes:
                if axis not in base_axes:  # colorbar etc.
                    axis.remove()
                else:
                    # subplot.cla()  # this also clears titles and subplot labels
                    for artist_list in [axis.lines, axis.patches, axis.texts, axis.tables, axis.artists, axis.images, axis.collections]:
                        try:
                            while artist_list:
                                artist_list[0].remove()
                        except AttributeError:
                            warnings.warn(f"Failed to remove Matplotlib list '{artist_list}'", RuntimeWarning)
                    box = Bbox(positions[axis])
                    axis.set_position(box, which='active')
                    axis.set_subplotspec(specs[axis])
                    # subplot.set_title(titles[subplot])
            # plt.tight_layout()
            plot_frame_function(fig, frame)

        return animation.FuncAnimation(fig, clear_and_plot, repeat=repeat, frames=frame_count, interval=interval)

    def finalize(self, figure):
        plt.tight_layout()  # because subplot titles can be added after figure creation

    def close(self, figure):
        if isinstance(figure, plt.Figure):
            plt.close(figure)
        elif isinstance(figure, animation.FuncAnimation):
            plt.close(figure._fig)

    def show(self, figure):
        if isinstance(figure, plt.Figure):
            figure.show()
        elif isinstance(figure, animation.FuncAnimation):
            if 'ipykernel' in sys.modules:
                from IPython.display import HTML
                return HTML(figure.to_html5_video())
            else:
                figure._fig.show()
        else:
            raise ValueError(f"{type(figure)} is not a valid {self.name} figure")

    def save(self, figure, path: str, dpi: float, transparent: bool):
        if isinstance(figure, plt.Figure):
            figure.savefig(path, dpi=dpi, transparent=transparent)
        elif isinstance(figure, animation.Animation):
            figure.save(path, dpi=dpi)
        else:
            raise ValueError(figure)


def _get_range(bounds: Box, index: int):
    lower = float(bounds.lower.vector[index].min)
    upper = float(bounds.upper.vector[index].max)
    if not math.is_finite(lower) and not math.is_finite(upper):
        lower = -.1
        upper = .1
    elif not math.is_finite(lower):
        lower = upper - .1
    elif not math.is_finite(upper):
        upper = lower + .1
    return lower if math.is_finite(lower) else None, upper if math.is_finite(upper) else None


def _next_line_color(axes: Axes, kind: str = None, get_index=False):
    kind = ['patches', 'lines', 'collections', 'containers'] if kind is None else kind.split(',')
    next_index = max([len(getattr(axes, k)) for k in kind])
    if get_index:
        return next_index
    return _default_color(next_index)


def _default_color(i: int):
    default_colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    return default_colors[i % len(default_colors)]


class LinePlot(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_grid and data.spatial_rank == 1 and not instance(data)

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        x = data.center[{'vector': 0, '~vector': 0}].numpy()
        requires_legend = False
        for c_idx, c_idx_n in zip(channel(data).meshgrid(), channel(data).meshgrid(names=True)):
            label = index_label(c_idx_n)
            values = data.values.vector.dual[0][c_idx].numpy()
            if (color[c_idx] == None).all:
                col = _next_line_color(subplot)
            else:
                col = _plt_col(color[c_idx])
            alpha_f = float(alpha[c_idx])
            if ((err[c_idx] != 0) & (err[c_idx] != None)).any:
                v_err = reshaped_numpy(err[c_idx], [spatial(data)])
                subplot.fill_between(x, values - v_err, values + v_err, color=col, alpha=alpha_f * .2)
            if values.dtype in (np.complex64, np.complex128):
                subplot.plot(x, values.real, label=f"{label} real" if label else "real", color=col, alpha=alpha_f)
                subplot.plot(x, values.imag, '--', label=f"{label} imag" if label else "imag", color=col, alpha=alpha_f)
                requires_legend = True
            else:
                subplot.plot(x, values, label=label, color=col, alpha=alpha_f)
                requires_legend = requires_legend or label
        if requires_legend:
            # if not has_legend_like([index_label(idx_n) for idx_n in channel(data.values).meshgrid(names=True)], figure):
            subplot.legend()
        # elif min_val is not None and max_val is not None:
        #     subplot.set_ylim((min_val - .02 * (max_val - min_val), max_val + .02 * (max_val - min_val)))
        if spatial(data).item_names[0]:  # label x ticks
            set_ticks(subplot, 0, x, spatial(data).item_names[0])


class BarChart(Recipe):

    def __init__(self, col_width=.8):
        self.col_width = col_width

    def can_plot(self, data: Field, space: Box) -> bool:
        return instance(data) and data.geometry.vector.size == 1 and data.geometry.vector.item_names == instance(data).names and spatial(data.values).is_empty

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        vector = data.geometry.shape['vector']
        x, = reshaped_numpy(data.points, [vector, instance(data)])
        if len(x) == 1:
            width = 1.
        else:
            x_range = x.max() - x.min()
            x_min = x.min() - x_range / (len(x)-1) / 2
            x_max = x.max() + x_range / (len(x)-1) / 2
            width = x_max - x_min
        channels = channel(data.values).volume
        for i, ch in enumerate(channel(data.values).meshgrid(names=True)):
            height = reshaped_numpy(data.values[ch], [instance(data)])
            errs = reshaped_numpy(err[ch], [instance(data)])
            cols = matplotlib_colors(color[ch], instance(data))
            alpha_f = float(alpha[ch].max)
            w = self.col_width / channels * width / len(x)
            pos = x + w * i + w/2 - w * channels / 2
            bar_plt = subplot.bar(pos, height=height, width=w, yerr=errs, color=cols, alpha=alpha_f, label=index_label(ch))
            if channels < 3:
                try:
                    subplot.bar_label(bar_plt, label_type='edge', fmt='%.2f' if data.values.dtype.kind == float else '%d')
                except AttributeError:
                    warnings.warn(f"Matplotlib is outdated, version={matplotlib.__version__}. Update it to show bar labels", RuntimeWarning)
        set_ticks(subplot, 0, x, instance(data).item_names[0])
        if channels > 1:
            if not has_legend_like([index_label(ch) for ch in channel(data.values).meshgrid(names=True)], figure):
                subplot.legend()
        if float(math.finite_min(data.values, shape)) > 0:
            subplot.set_ylim((0, float(math.finite_max(data.values, shape)) * 1.1))


class Histogram(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_point_cloud and data.geometry.vector.size == 1 and data.geometry.vector.item_names == spatial(data).names and spatial(data.values).rank == 1 and not instance(data)

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        orientation = ['vertical', 'horizontal'][space.vector.item_names.index(data.geometry.vector.item_names[0])]
        vector = data.bounds.shape['vector']
        x, = reshaped_numpy(data.points, [vector, spatial(data)])
        bin_edges = [x[0] - (x[1]-x[0]) / 2, *((x[1:] + x[:-1]) / 2), x[-1] + (x[-1] - x[-2]) / 2]
        for i, ch in enumerate(channel(data.values).meshgrid(names=True)):
            line_width = 1.5 + .5 * (channel(data.values).volume - i - 1)
            counts = reshaped_numpy(data.values[ch], [spatial(data)])
            # errs = reshaped_numpy(err[ch], [spatial(data)])  ToDo
            histtype = 'bar' if i == 0 else 'step'
            if (color[ch] == None).all:
                col = _default_color(i)
            else:
                col = _plt_col(color[ch])
            alpha_fac = .3 if i == 0 else 1
            subplot.hist(bin_edges[:-1], bins=bin_edges, weights=counts, orientation=orientation, histtype=histtype, color=col, alpha=float(alpha[ch].max) * alpha_fac, label=index_label(ch), linewidth=line_width)
        if channel(data.values).volume > 1:
            if not has_legend_like([index_label(idx_n) for idx_n in channel(data.values).meshgrid(names=True)], figure):
                subplot.legend()


class Heatmap2D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_grid and channel(data).volume == 1 and data.spatial_rank == 2 and not instance(data) and math.is_finite(data.values).any

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = spatial(data)
        vector = data.geometry.shape['vector']
        bounds = data.geometry.bounds
        if bounds.upper.vector.item_names is not None:
            left, bottom = bounds.lower.vector[dims]
            right, top = bounds.upper.vector[dims]
        else:
            dim_indices = data.resolution.indices(dims)
            left, bottom = bounds.lower.vector[dim_indices]
            right, top = bounds.upper.vector[dim_indices]
        extent = (float(left), float(right), float(bottom), float(top))
        if space.spatial_rank == 3:  # surface plot
            z = data.values.numpy(dims)
            x, y = reshaped_numpy(data.points, [vector, *spatial(data)])
            plot_surface(subplot, x, y, z)
        else:  # heatmap
            aspect = subplot.get_aspect()
            image = data.values.numpy(dims.reversed)
            if data.values.dtype.kind == complex:
                amplitude = abs(image) / np.max(abs(image))
                phase = np.angle(image) / (2*np.pi) + .5
                hsv = np.stack([phase, np.ones_like(amplitude), amplitude], -1)
                rgb = matplotlib.colors.hsv_to_rgb(hsv)
                subplot.imshow(rgb, origin='lower', extent=extent, vmin=min_val, vmax=max_val, aspect=aspect, alpha=float(alpha))
            else:
                subplot.imshow(image, origin='lower', extent=extent, vmin=min_val, vmax=max_val, aspect=aspect, alpha=float(alpha))
        if show_color_bar:
            add_color_bar(subplot, data.values.numpy(dims), min_val, max_val)
        return True


class VectorField2D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_grid and data.spatial_rank == 2 and 'vector' in data.shape

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        vector = data.geometry.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        data = data.at_centers()
        x, y = reshaped_numpy(data.points.vector[dims], [vector, data.shape.without('vector')])
        u, v = reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.without('vector')])
        alphas = reshaped_numpy(alpha, [vector, data.shape.without('vector')])
        colors = matplotlib_colors(color, data.shape.without('vector'))
        for ch in range(u.shape[0]):
            subplot.quiver(x, y, u[ch], v[ch], color=colors, alpha=alphas[ch], units='xy', scale=1)
        return True


class VectorField3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_grid and channel(data).volume > 1 and data.spatial_rank == 3

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        vector = data.geometry.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        data = data.at_centers()
        x, y, z = reshaped_numpy(data.points.vector[dims], [vector, data.shape.without('vector')])
        u, v, w = reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.without('vector')])
        alphas = reshaped_numpy(alpha, [vector, data.shape.without('vector')])
        colors = matplotlib_colors(color, data.shape.without('vector'))
        for ch in range(u.shape[0]):
            subplot.quiver(x, y, z, u[ch], v[ch], w[ch], color=colors, alpha=alphas[ch])


class Heatmap3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_grid and channel(data).volume == 1 and data.spatial_rank == 3

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        xyz = StaggeredGrid(lambda x: x, math.extrapolation.BOUNDARY, data.geometry.bounds, data.resolution).staggered_tensor().numpy(dims + ('vector',))[:-1, :-1, :-1, :]
        xyz = xyz.reshape(-1, 3)
        values = data.values.numpy(dims).flatten()
        if wrap(color == 'cmap').all:
            color = 0
        col = matplotlib.colors.to_rgba(_plt_col(color))
        colors = np.zeros_like(values)[..., None] + col
        norm_values = (values - min_val) / (max_val - min_val)
        size = len(values) ** .3333
        exponent = size / 15
        alpha = float(alpha) * norm_values ** exponent
        colors[..., -1] *= alpha
        size = [data.dx.numpy()] * len(xyz)
        cubes = plotCubeAt(xyz, size, colors)
        subplot.add_collection3d(cubes)


def cuboid_data(min_xyz: np.ndarray, size=(1, 1, 1)):
    x = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    x = np.array(x).astype(float) * size
    return x + min_xyz


def plotCubeAt(positions, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)):
        colors = ["C0"] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)):
        sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data(p, size=s))
    return Poly3DCollection(np.concatenate(g), facecolors=np.repeat(colors, 6, axis=0), **kwargs)


class VectorCloud2D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 2 and 'vector' in channel(data)

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        vector = data.geometry.shape['vector']
        channels = channel(data).without('vector')
        data = only_stored_elements(data)
        for idx in channels.meshgrid(names=True):
            c_data = data[idx]
            x, y = reshaped_numpy(c_data.center[dims], [vector, c_data.shape.without('vector')])
            u, v = reshaped_numpy(c_data.values.vector[dims], [vector, c_data.shape.without('vector')])
            color_i = color[idx]
            if wrap(color[idx] == 'cmap').all:
                col = _next_line_color(subplot, kind='collections')  # ToDo
            elif color[idx].shape:
                col = [_plt_col(c) for c in color_i.numpy(c_data.shape.non_channel).reshape(-1)]
            else:
                col = _plt_col(color[idx])
            alphas = reshaped_numpy(alpha, [c_data.shape.without('vector')])
            subplot.quiver(x, y, u, v, color=col, units='xy', scale=1, alpha=alphas, label=index_label(idx) if channels.volume > 1 else None)
        if channels and not has_legend_like([index_label(idx_n) for idx_n in channels.meshgrid(names=True)], figure):
            subplot.legend()


class VectorCloud3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 3 and 'vector' in channel(data)

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        vector = data.geometry.shape['vector']
        x, y, z = reshaped_numpy(data.center[dims], [vector, data.shape.without('vector')])
        u, v, w = reshaped_numpy(data.values[dims], [vector, data.shape.without('vector')])
        if color.shape:
            col = [_plt_col(c) for c in color.numpy(data.shape.non_channel).reshape(-1)]
        else:
            col = _plt_col(color)
        alphas = reshaped_numpy(alpha, [data.shape.without('vector')])
        subplot.quiver(x, y, z, u, v, w, color=col, alpha=alphas)


class StreamPlot2D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 2 and 'vector' in channel(data) and data.is_grid and (data.values != 0).any and all(dim.size > 1 for dim in data.resolution)

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        vector = data.geometry.shape['vector']
        data = data.at_centers()
        with math.precision(64):  # streamplot requires very precise grid spacing
            x, y = reshaped_numpy(data.points, [vector, *data.shape.without('vector')])
        x = x[:, 0]
        y = y[0, :]
        u, v = reshaped_numpy(data.values.vector[vector.item_names[0]], [vector, *data.shape.without('vector')])
        if wrap(color == 'cmap').all:
            col = reshaped_numpy(math.vec_length(data.values), [*data.shape.without('vector')]).T
        elif color.shape:
            col = [_plt_col(c) for c in color.numpy(data.shape.non_channel).reshape(-1)]
        else:
            col = _plt_col(color)
        alphas = reshaped_numpy(alpha, [data.shape.without('vector')])
        a = float(alphas[0])
        prev_patches = set(subplot.patches)
        try:
            stream = subplot.streamplot(x, y, u.T, v.T, color=col, cmap=colormaps.get_cmap(matplotlib.rcParams['image.cmap']))
        except ValueError as err:  # no lines cause
            if err.args[0] == "need at least one array to concatenate":
                return
            raise err
        stream.lines.set_alpha(a)
        new_patches = set(subplot.patches) - prev_patches
        for obj in new_patches:
            # if isinstance(obj, matplotlib.patches.FancyArrowPatch):
            obj.set_alpha(a)


class EmbeddedPoint2D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 1 and space.spatial_rank == 2

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        present_dim = data.geometry.vector.item_names[0]
        assert present_dim in space.vector.item_names
        horizontal = present_dim == space.vector.item_names[1]
        if data.geometry.bounding_radius().max == 0:
            if horizontal:
                x = [float(space.lower.vector[0]), float(space.upper.vector[0])]
                y, = reshaped_numpy(data.points, ['vector', instance])
                subplot.plot(x, [y, y], '--', color='grey')
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("Range not yet supported")


class Heightmap2D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 2 and isinstance(data.geometry, Heightmap)

    @math.broadcast(dims=instance, unwrap_scalars=False)
    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        heightmap: Heightmap = data.geometry
        x, y = reshaped_numpy(heightmap.vertices[dims], ['vector', *spatial(heightmap.height)])
        if (color == None).all:
            col = _next_line_color(subplot)
        else:
            col = _plt_col(color)
        alpha_f = float(alpha)
        if heightmap._hdim == dims[1]:  # horizontal
            if heightmap._fill_below:
                y1, y2 = max(-1e10, float(heightmap.bounds[heightmap._hdim].lower)), y
            else:
                y1, y2 = y, min(1e10, float(heightmap.bounds[heightmap._hdim].upper))
            subplot.fill_between(x, y1, y2, color=col, alpha=alpha_f)
        else:
            if heightmap._fill_below:
                x1, x2 = max(-1e10, float(heightmap.bounds[heightmap._hdim].lower)), x
            else:
                x1, x2 = x, min(1e10, float(heightmap.bounds[heightmap._hdim].upper))
            subplot.fill_betweenx(y, x1, x2)


class Heightmap3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 3 and isinstance(data.geometry, Heightmap)

    @math.broadcast(dims=instance, unwrap_scalars=False)
    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        heightmap: Heightmap = data.geometry
        x, y, z = reshaped_numpy(heightmap.vertices[dims], ['vector', *spatial(heightmap.height)])
        if (color == None).all:
            col = _next_line_color(subplot)
        else:
            col = _plt_col(color)
        alpha_f = float(alpha)
        plot_surface(subplot, x, y, z, color=col, alpha=alpha_f)


class PointCloud2D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.spatial_rank == 2

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        vector = data.geometry.shape['vector']
        channels = channel(data.points).without('vector')
        legend_patches = []
        # if (color == None).all:
        #     color = math.range_tensor(channels)
        for idx, idx_n in zip(channels.meshgrid(), channels.meshgrid(names=True)):
            col = color[idx]
            PointCloud2D._plot_points(subplot, data[idx], dims, vector, col, alpha[idx], err[idx], min_val, max_val, index_label(idx_n))
            if col.rank < color.rank or ((color == None).all and channels.volume > 1):  # There are multiple colors
                legend_patches = True
        if legend_patches:
            if not has_legend_like([index_label(idx_n) for idx_n in channels.meshgrid(names=True)], figure):
                subplot.legend()

    @staticmethod
    def _plot_points(axis: Axes, data: Field, dims: tuple, vector: Shape, color: Tensor, alpha: Tensor, err: Tensor, min_val, max_val, label):
        connected = spatial(data.points)
        if isinstance(data.sampled_elements, GeometryStack):
            if data.sampled_elements.is_union:
                for idx in data.sampled_elements.object_dims.meshgrid():
                    PointCloud2D._plot_points(axis, data[idx], dims, vector, color[idx], alpha[idx], err[idx], min_val, max_val, label)
                return
            elif data.sampled_elements.is_intersection:
                math.assert_close(data.values, math.NAN, msg="Intersections can only be plotted as Geometries, not Fields.")
                sdf = CenteredGrid(data.sampled_elements.approximate_signed_distance, bounds=data.sampled_elements.bounding_box().scaled(1.).corner_representation(), **{d: 32 for d in dims})
                sdf_grid = SDFGrid(sdf.values, sdf.bounds, approximate_outside=False)
                data = Field(sdf_grid, math.NAN, 0)
        data = only_stored_elements(data)
        x, y = reshaped_numpy(data.points.vector[dims], ['vector', non_channel(data)])
        if wrap(color == 'cmap').all:
            values = reshaped_numpy(data.values, [non_channel(data)])
            mpl_colors = add_color_bar(axis, values, min_val, max_val)
            single_color = False
        elif non_channel(data).only(color.shape) and color.dtype.kind == float:  # use color map
            values = reshaped_numpy(color, [non_channel(data)])
            mpl_colors = add_color_bar(axis, values, None, None)
            single_color = False
        else:
            mpl_colors = matplotlib_colors(color, non_channel(data), default=0)
            single_color = True
        alphas = reshaped_numpy(alpha, [non_channel(data)])
        if isinstance(data.geometry, Point):
            if spatial(data.points).is_empty:
                axis.scatter(x, y, color=mpl_colors, s=6 ** 2, alpha=alphas)
                if (err != 0).any:
                    x_err = reshaped_numpy(err.vector[dims[0]], [instance(data)]) if dims[0] in err.vector.item_names else 0
                    y_err = reshaped_numpy(err.vector[dims[1]], [instance(data)]) if dims[1] in err.vector.item_names else 0
                    if all(np.all(c == mpl_colors[0]) for c in mpl_colors) and all(a == alphas[0] for a in alphas):
                        axis.errorbar(x, y, y_err, x_err, fmt=' ', color=mpl_colors[0], alpha=alphas[0])
                    else:
                        for x_, y_, y_err_, x_err_, col_, alpha_ in zip(x, y, y_err, x_err, mpl_colors, alphas):
                            axis.errorbar(x_, y_, y_err_, x_err_, fmt=' ', color=col_, alpha=alpha_)
        else:
            if isinstance(data.geometry, Sphere):
                rad = reshaped_numpy(data.geometry.bounding_radius(), [data.shape.non_channel])
                shapes = [plt.Circle((xi, yi), radius=ri, linewidth=0, alpha=a, facecolor=ci) for xi, yi, ri, ci, a in zip(x, y, rad, mpl_colors, alphas)]
                axis.add_collection(matplotlib.collections.PatchCollection(shapes, match_original=True))
            elif isinstance(data.geometry, BaseBox):
                half_size = data.geometry.half_size
                min_len = axis.get_ylim()[1] - axis.get_ylim()[0] + axis.get_xlim()[1] - axis.get_xlim()[0]
                half_size = math.where(math.is_finite(half_size), half_size, min_len)
                w2, h2 = reshaped_numpy(half_size, ['vector', data.shape.non_channel])
                if data.geometry.rotation_matrix is None:
                    angles = w2 * 0.
                    lower_x = x - w2
                    lower_y = y - h2
                else:
                    angles = reshaped_numpy(rotation_angles(data.geometry.rotation_matrix), [data.shape.non_channel])
                    lower_x, lower_y = reshaped_numpy(data.geometry.center - rotate(data.geometry.half_size, data.geometry.rotation_matrix), ['vector', data.shape.non_channel])
                shapes = [plt.Rectangle((lxi, lyi), w2i * 2, h2i * 2, angle=ang*180/np.pi, linewidth=1, edgecolor='white', alpha=a, facecolor=ci) for lxi, lyi, w2i, h2i, ang, ci, a in zip(lower_x, lower_y, w2, h2, angles, mpl_colors, alphas)]
                axis.add_collection(matplotlib.collections.PatchCollection(shapes, match_original=True))
            elif isinstance(data.geometry, Mesh):
                edgecolor = 'white' if single_color else None
                csr = data.mesh.elements.numpy().tocsr()
                xy = reshaped_numpy(data.geometry.vertices.center, [instance, 'vector'])
                shapes = []
                for i in range(instance(data).volume):
                    vert_indices = csr.indices[csr.indptr[i]:csr.indptr[i+1]]
                    xyi = xy[vert_indices]
                    shapes.append(plt.Polygon(xyi, closed=True, edgecolor=edgecolor, alpha=alphas[i], facecolor=mpl_colors[i]))
                axis.add_collection(matplotlib.collections.PatchCollection(shapes, match_original=True))
            elif isinstance(data.geometry, Graph):
                xs, ys = reshaped_numpy(data.geometry.center, ['vector', non_channel])
                if isinstance(data.graph.nodes, Point):
                    axis.scatter(xs, ys, color=mpl_colors, alpha=alphas)
                else:
                    PointCloud2D._plot_points(axis, to_field(data.graph.nodes), dims, vector, color, alpha, err, min_val, max_val, label)
                if math.is_sparse(data.graph.edges):
                    edges = math.stored_indices(data.graph.edges)
                    edge_val = math.to_float(math.stored_values(data.graph.edges))
                    if channel(edge_val):
                        edge_val = math.unstack(edge_val, channel)[0]
                    edges = edges[edge_val != 0]
                    edge_val = edge_val[edge_val != 0]
                else:
                    edges = math.nonzero(data.graph.connectivity, index_dim=channel('index'))
                    edge_val = data.graph.edges[edges]
                p1, p2 = edges.index
                x1, y1 = reshaped_numpy(data.graph.center[p1], ['vector', instance])
                x2, y2 = reshaped_numpy(data.graph.center[p2], ['vector', instance])
                if wrap(color == 'cmap').all:
                    edge_val = reshaped_numpy(edge_val, [instance])
                    edge_colors = add_color_bar(axis, edge_val, min_val, max_val)
                    if edge_val.min() == edge_val.max():
                        edge_colors = edge_colors[0]
                else:
                    edge_colors = mpl_colors[0]
                if np.array(edge_colors).ndim <= 1:
                    axis.plot([x1, x2], [y1, y2], color=edge_colors, alpha=alphas[0])
                else:
                    for i, (x1_, x2_, y1_, y2_, col) in enumerate(zip(x1, x2, y1, y2, edge_colors)):
                        axis.plot([x1_, x2_], [y1_, y2_], color=col, alpha=alphas[0])
            elif isinstance(data.geometry, SDFGrid):
                d = data.geometry.values.numpy(dims)
                x, y = data.geometry.points.numpy(('vector',) + dims)
                x = x[:, 0]
                y = y[0, :]
                axis.contourf(x, y, d.T, levels=[float('-inf'), 0], colors=[mpl_colors[0]], alpha=alphas[0])
            elif isinstance(data.geometry, SDF):
                bounds = data.geometry.bounding_box()
                grid = UniformGrid(spatial(**{d: 100 for d in dims}), bounds=bounds).center
                sdf = data.geometry(grid)
                d = sdf.numpy(dims)
                x, y = grid.numpy(('vector',) + dims)
                x = x[:, 0]
                y = y[0, :]
                axis.contourf(x, y, d.T, levels=[float('-inf'), 0], colors=[mpl_colors[0]], alpha=alphas[0])
            else:
                rad = reshaped_numpy(data.geometry.bounding_radius(), [data.shape.non_channel])
                shapes = [plt.Circle((xi, yi), radius=ri, linewidth=0, alpha=a, facecolor=ci) for xi, yi, ri, ci, a in zip(x, y, rad, mpl_colors, alphas)]
                axis.add_collection(matplotlib.collections.PatchCollection(shapes, match_original=True))
        if connected:  # Connect by line
            for i, idx in enumerate(instance(data).meshgrid()):
                for sp_dim in spatial(data):
                    other_sp = spatial(data).without(sp_dim)
                    xs, ys = reshaped_numpy(data[idx].points.vector[dims], [vector, sp_dim, other_sp])
                    if (color == None).all:
                        col = _next_line_color(axis, 'lines' if connected else 'collections')
                    elif non_channel(data).only(color.shape) and color.dtype.kind == float:  # use color map
                        if non_channel(data).after_gather(idx).only(color.shape):
                            values = reshaped_numpy(color[idx], [non_channel(data).after_gather(idx)])
                            col = add_color_bar(axis, values, color.min, color.max)
                            col = col[0]  # ToDo call plot() for each line segment :-(
                            warnings.warn("Changing color by line segment not yet supported", RuntimeWarning)
                        else:
                            values = reshaped_numpy(color[idx], [])
                            col = add_color_bar(axis, values, color.min, color.max)
                    else:
                        col = _plt_col(color)
                    alpha_f = float(alpha[idx].max)
                    if ((err[idx] != 0) & (err[idx] != None)).any:
                        x_errs = reshaped_numpy(err[idx].vector[dims[0]], [other_sp, sp_dim]) if dims[0] in err.vector.item_names else 0
                        y_errs = reshaped_numpy(err[idx].vector[dims[1]], [other_sp, sp_dim]) if dims[1] in err.vector.item_names else 0
                        for x, y, x_err, y_err in zip(xs.T, ys.T, x_errs, y_errs):
                            if math.max(x_err) > math.max(y_err):
                                axis.fill_betweenx(y, x - x_err, x + x_err, color=col, alpha=alpha_f * .2)
                            else:
                                axis.fill_between(x, y - y_err, y + y_err, color=col, alpha=alpha_f * .2)
                    axis.plot(xs, ys, color=col, alpha=alpha_f, label=label if i == 0 else None)
            if isinstance(data.geometry, Point) and 2 < spatial(data.geometry).volume < 100:  # plot small dots on lines
                xs, ys = reshaped_numpy(data.geometry.center, ['vector', non_channel])
                axis.scatter(xs, ys, s=3, marker='o', c=mpl_colors, alpha=alphas)

        if any(non_channel(data).item_names):
            PointCloud2D._annotate_points(axis, data.points, color, alpha, dims)

    @staticmethod
    def _annotate_points(axis, points: math.Tensor, color: Tensor, alpha: Tensor, dims: Tuple[str], label_axis=True, max_axis_labels=10):
        labeled_dims = non_channel(points)
        labeled_dims = math.concat_shapes(*[d for d in labeled_dims if d.item_names[0]])
        if not labeled_dims:
            return
        if all(dim.name in points.shape.get_item_names('vector') for dim in labeled_dims):
            if label_axis:
                for labeled_dim in labeled_dims:
                    if len(labeled_dim.item_names[0]) <= max_axis_labels:
                        if points.vector[labeled_dim.name].shape.without(labeled_dims).volume > 1:
                            return  # we'd have to duplicate names
                        which_axis = dims.index(labeled_dim.name)
                        set_ticks(axis, which_axis, reshaped_numpy(points.vector[labeled_dim.name], [shape]))
            return  # The point labels match one of the figure axes, so they are redundant
        if points.shape['vector'].size == 2:
            np_points = points.numpy([..., 'vector'])
            rel_pos = axis.transAxes.inverted().transform(axis.transData.transform(np_points))
            x_view = axis.get_xlim()[1] - axis.get_xlim()[0]
            y_view = axis.get_ylim()[1] - axis.get_ylim()[0]
            for (x, y), (rx, ry), idx, idx_n in zip(np_points, rel_pos, labeled_dims.meshgrid(), labeled_dims.meshgrid(names=True)):
                horizontal_align = 'right' if rx >= .5 else 'left'
                if axis.get_xscale() == 'log':
                    offset_x = x * (1 + .0003 * x_view) if rx < .5 else x * (1 - .0003 * x_view)
                else:
                    offset_x = x + .01 * x_view if rx < .5 else x - .01 * x_view
                if axis.get_yscale() == 'log':
                    offset_y = y * (1 + .0003 * y_view) if ry < .5 else y * (1 - .0003 * y_view)
                else:
                    offset_y = y + .01 * y_view if ry < .5 else y - .01 * y_view
                axis.text(offset_x, offset_y, index_label(idx_n), color=_plt_col(color[idx]), alpha=float(alpha[idx]), ha=horizontal_align)


class PointCloud3D(Recipe):

    def can_plot(self, data: Field, space: Box) -> bool:
        return data.is_point_cloud and data.spatial_rank == 3

    def plot(self, data: Field, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        vector = data.geometry.shape['vector']
        channels = channel(data.points).without('vector')
        for idx in channels.meshgrid(names=True):
            x, y, z = reshaped_numpy(data[idx].points.vector[dims], [vector, non_channel(data)])
            mpl_colors = matplotlib_colors(color[idx], non_channel(data), default=0)
            alphas = reshaped_numpy(alpha[idx], [non_channel(data)])
            M = subplot.transData.get_matrix()
            x_scale, y_scale, z_scale = M[0, 0], M[1, 1], M[2, 2]
            if isinstance(data.geometry, Sphere):
                rx, ry, rz = reshaped_numpy(data.geometry.radius, [vector, data.geometry.shape.without('vector')])
                u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)  # Set of all spherical angles
                # --- Cartesian coordinates that correspond to the spherical angles ---
                for i in range(len(x)):
                    x_ = x[i] + rx[i] * np.outer(np.cos(u), np.sin(v))
                    y_ = y[i] + ry[i] * np.outer(np.sin(u), np.sin(v))
                    z_ = z[i] + rz[i] * np.outer(np.ones_like(u), np.cos(v))
                    plot_surface(subplot, x_, y_, z_, rstride=4, cstride=4, color=mpl_colors[0], alpha=alphas[0])
            elif isinstance(data.geometry, BaseBox):
                a = alphas[0]
                c = mpl_colors[0]
                # ToDo support collections of boxes
                cx, cy, cz = math.reshaped_numpy(data.geometry.corners, ['vector', *['~'+d for d in dims]])
                plot_surface(subplot, cx[:, :, 1], cy[:, :, 1], cz[:, :, 1], alpha=a, color=c)
                plot_surface(subplot, cx[:, :, 0], cy[:, :, 0], cz[:, :, 0], alpha=a, color=c)
                plot_surface(subplot, cx[:, 1, :], cy[:, 1, :], cz[:, 1, :], alpha=a, color=c)
                plot_surface(subplot, cx[:, 0, :], cy[:, 0, :], cz[:, 0, :], alpha=a, color=c)
                plot_surface(subplot, cx[1, :, :], cy[1, :, :], cz[1, :, :], alpha=a, color=c)
                plot_surface(subplot, cx[0, :, :], cy[0, :, :], cz[0, :, :], alpha=a, color=c)
            elif isinstance(data.geometry, Point):
                if not spatial(data.geometry):
                    size = 6 / (0.5 * (x_scale+y_scale+z_scale)/3)
                    subplot.scatter(x, y, z, color=mpl_colors, alpha=alphas, s=(size * 0.5 * (x_scale + y_scale + z_scale) / 3) ** 2)
            elif isinstance(data.geometry, _EmbeddedGeometry):
                raise NotImplementedError(f"Plotting embedded geometries not yet supported")
            else:
                size = data.geometry.bounding_radius().numpy()
                subplot.scatter(x, y, z, marker='X', color=mpl_colors, alpha=alphas, s=(size * 0.5 * (x_scale + y_scale + z_scale) / 3) ** 2)
            if spatial(data.geometry):  # Connect by lines
                for i, idx in enumerate(instance(data).meshgrid()):
                    for sp_dim in spatial(data.geometry):
                        other_sp = spatial(data.geometry).without(sp_dim)
                        xs, ys, zs = reshaped_numpy(data[idx].points.vector[dims], [vector, sp_dim, other_sp])
                        if (color == None).all:
                            col = _next_line_color(subplot)
                        else:
                            col = _plt_col(color)
                        alpha_f = float(alpha[idx].max)
                        if ((err[idx] != 0) & (err[idx] != None)).any:
                            x_errs = reshaped_numpy(err[idx].vector[dims[0]], [other_sp, sp_dim]) if dims[0] in err.vector.item_names else 0
                            y_errs = reshaped_numpy(err[idx].vector[dims[1]], [other_sp, sp_dim]) if dims[1] in err.vector.item_names else 0
                            for x, y, x_err, y_err in zip(xs.T, ys.T, x_errs, y_errs):
                                if math.max(x_err) > math.max(y_err):
                                    subplot.fill_betweenx(y, x - x_err, x + x_err, color=col, alpha=alpha_f * .2)
                                else:
                                    subplot.fill_between(x, y - y_err, y + y_err, color=col, alpha=alpha_f * .2)
                        for i in range(xs.shape[1]):
                            subplot.plot(xs[:, i], ys[:, i], zs[:, i], color=col, alpha=alpha_f)
                            if isinstance(data.geometry, Point) and 2 < spatial(data.geometry).volume < 100:
                                subplot.scatter(xs[:, i], ys[:, i], zs[:, i], s=3, marker='o', c=col, alpha=alphas)


def _plt_col(col):
    if isinstance(col, Tensor):
        col = next(iter(col))
    if col is None:
        cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        return cycle[0]
    if not isinstance(col, (str, tuple, list)):
        cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        col = cycle[int(col) % len(cycle)]
        return col
    if isinstance(col, str) and col.startswith('#'):
        return col
        # col = tuple(int(col.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    col = np.asarray(col)
    if col.dtype.kind == 'i':
        col = col / 255.
    return col


def matplotlib_colors(color: Tensor, dims: Shape, default=None) -> Union[list, None]:
    if color.rank == 0 and wrap(color == 'cmap').all:
        if default is None:
            return None
        else:
            color = math.wrap(default)
    color = color[shape(color).without(dims).first_index()]  # Select first color along unlisted dimensions
    if color.dtype.kind == int:
        cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        return [cycle[int(color[idx]) % len(cycle)] for idx in dims.meshgrid()]
    else:
        return [color[idx].native() for idx in dims.meshgrid()]


def add_color_bar(axis: Axes, values, min_val, max_val, cmap=None):
    if cmap is None:
        cmap = colormaps.get_cmap(matplotlib.rcParams['image.cmap'])
    figure = axis.figure
    figure_has_color_bar = any(['colorbar' in ax.get_label() for ax in figure.axes])
    is_complex = np.iscomplex(max_val)
    if min_val is None or max_val is None or not figure_has_color_bar:
        if min_val is not None and max_val is not None:  # only one color bar for all subplots
            figure.subplots_adjust(left=.1, bottom=.1, right=0.85, top=.92)
            cax = figure.add_axes([0.87, 0.1, 0.08 if is_complex else 0.03, 0.82])
        else:
            cax = None
        if is_complex:
            if cax is not None:
                amp, phase = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
                cbar_img = matplotlib.colors.hsv_to_rgb(np.stack([phase, np.ones_like(phase), amp], -1))
                cmap_extent = (min_val, max_val, 0, 2 * np.pi)
                cax.imshow(cbar_img, extent=cmap_extent, aspect='auto', origin='lower')
                cax.yaxis.tick_right()
                cax.set_yticks([0, np.pi, 2 * np.pi], ['0', 'π', '2π'])
                cax.yaxis.set_label_position('right')
        else:
            min_val = np.min(values) if min_val is None else min_val
            max_val = np.max(values) if max_val is None else max_val
            norm = Normalize(vmin=min_val, vmax=max_val)
            mappable = ScalarMappable(norm=norm, cmap=cmap)
            figure.colorbar(mappable, ax=axis if cax is None else None, cax=cax)  # adds a new Axis to the figure
    else:
        assert figure_has_color_bar
        # assume same min/max
        norm = Normalize(vmin=min_val, vmax=max_val)
    return cmap(norm(values))


def plot_surface(axes, *args, **kwargs):
    if hasattr(axes, '_phi_z_order_index'):
        idx = axes._phi_z_order_index
        axes.plot_surface(*args, zorder=.1 + idx * 1e-3, **kwargs)
        axes._phi_z_order_index += 1
    else:
        axes.plot_surface(*args, **kwargs)


def has_legend_like(labels, figure):
    for axes in figure.axes:
        if axes.legend_ is not None:
            texts = [t._text for t in axes.legend_.texts]
            if texts == list(labels):
                return True
    return False


def set_ticks(axes, which_axis, values, names=None):
    values = sorted(set(values))
    if which_axis == 0:  # x axis
        if hasattr(axes, '_xticks_from_data'):
            values = list(sorted(set([*values, *axes.get_xticks()])))
        axes.set_xticks(values, names)
        if axes.get_xscale() == 'log':
            axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes._xticks_from_data = True
    elif which_axis == 1:
        if hasattr(axes, '_yticks_from_data'):
            values = list(sorted(set([*values, *axes.get_yticks()])))
        axes.set_yticks(values, names)
        if axes.get_yscale() == 'log':
            axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes._yticks_from_data = True


def _get_pixels_per_unit(fig: plt.Figure, axis: plt.Axes, dpi=90):
    M = axis.transData.get_matrix()
    x_scale, y_scale = M[0, 0], M[1, 1]  # fig_size_px/unit
    # subplot_width = subplot.figbox.width * subplot.figure.bbox_inches.width
    # subplot_height = subplot.figbox.height * subplot.figure.bbox_inches.height
    # units_x = subplot.get_xlim()[1] - subplot.get_xlim()[0]
    # units_y = subplot.get_ylim()[1] - subplot.get_ylim()[0]
    # result_x = subplot_width * dpi / units_x
    # result_y = subplot_height * dpi / units_y
    return min(x_scale, y_scale)


def savefig(filename: str, transparent=True):
    plt.savefig(filename, transparent=transparent)


MATPLOTLIB = MatplotlibPlots()
MATPLOTLIB.recipes.extend([
            LinePlot(),
            BarChart(),
            Histogram(),
            # --- 2D ---
            Heatmap2D(),
            StreamPlot2D(),
            VectorField2D(),
            VectorCloud2D(),
            Heightmap2D(),
            PointCloud2D(),
            EmbeddedPoint2D(),
            # Mesh2D(),
            # --- 3D ---
            VectorField3D(),
            Heatmap3D(),
            Heightmap3D(),
            VectorCloud3D(),
            PointCloud3D(),
        ])
