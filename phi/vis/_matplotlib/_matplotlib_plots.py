import sys
import warnings
from typing import Callable, Tuple, Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import rc
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from phi import math, field
from phi.field import Grid, StaggeredGrid, PointCloud, SampledField
from phi.geom import Sphere, BaseBox, Point, Box
from phi.geom._stack import GeometryStack
from phi.math import Tensor, channel, spatial, instance, non_channel, Shape
from phi.vis._vis_base import display_name, PlottingLibrary, Recipe, index_label


class MatplotlibPlots(PlottingLibrary):

    def __init__(self):
        super().__init__('matplotlib', [plt.Figure, animation.Animation])

    def create_figure(self,
                      size: tuple,
                      rows: int,
                      cols: int,
                      spaces: Dict[Tuple[int, int], Box],
                      titles: Dict[Tuple[int, int], str],
                      log_dims: Tuple[str, ...]) -> Tuple[Any, Dict[Tuple[int, int], Any]]:
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
                    if bounds.spatial_rank == 1:
                        axis.set_xlabel(display_name(bounds.vector.item_names[0]))
                        axis.set_xlim(_get_range(bounds, 0))
                        if bounds.vector.item_names[0] in log_dims:
                            axis.set_xscale('log')
                        if '_' in log_dims:
                            axis.set_yscale('log')
                    elif bounds.spatial_rank == 2:
                        axis.set_xlabel(display_name(bounds.vector.item_names[0]))
                        axis.set_ylabel(display_name(bounds.vector.item_names[1]))
                        x_range, y_range = [_get_range(bounds, i) for i in (0, 1)]
                        axis.set_xlim(x_range)
                        axis.set_ylim(y_range)
                        x_size, y_size = x_range[1] - x_range[0], y_range[1] - y_range[0]
                        any_log = False
                        if bounds.vector.item_names[0] in log_dims:
                            axis.set_xscale('log')
                            any_log = True
                        if bounds.vector.item_names[1] in log_dims:
                            axis.set_yscale('log')
                            any_log = True
                        if not any_log and x_size > 0 and y_size > 0 and max(x_size/y_size/subplot_aspect, y_size/x_size*subplot_aspect) < 4:
                            axis.set_aspect('equal', adjustable='box')
                    elif bounds.spatial_rank == 3:
                        axis.remove()
                        axis = axes[row, col] = figure.add_subplot(rows, cols, cols*row + col + 1, projection='3d')
                        axis.set_xlabel(display_name(bounds.vector.item_names[0]))
                        axis.set_ylabel(display_name(bounds.vector.item_names[1]))
                        axis.set_zlabel(display_name(bounds.vector.item_names[2]))
                        axis.set_xlim(_get_range(bounds, 0))
                        axis.set_ylim(_get_range(bounds, 1))
                        axis.set_zlim(_get_range(bounds, 2))
                        if bounds.vector.item_names[0] in log_dims:
                            warnings.warn("Only z axis can be log scaled in 3D plot with Matplotlib. Please reorder the dimensions.", RuntimeWarning)
                            # subplot.set_xscale('log')
                        if bounds.vector.item_names[1] in log_dims:
                            warnings.warn("Only z axis can be log scaled in 3D plot with Matplotlib. Please reorder the dimensions.", RuntimeWarning)
                            # subplot.set_yscale('log')
                        if bounds.vector.item_names[2] in log_dims:
                            axis.set_zscale('log')
                    axis.set_title(titles.get((row, col), None))
                    axes_by_pos[(row, col)] = axes[row, col]
        return figure, axes_by_pos

    def animate(self, fig: plt.Figure, frames: int, plot_frame_function: Callable, interval: float, repeat: bool):
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
                    try:
                        raise AttributeError
                        axis.lines.clear()
                        axis.patches.clear()
                        axis.texts.clear()
                        axis.tables.clear()
                        axis.artists.clear()
                        axis.images.clear()
                        axis.collections.clear()
                    except AttributeError:  # newer Matplotlib versions don't support clear() anymore
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
            plot_frame_function(frame)

        return animation.FuncAnimation(fig, clear_and_plot, repeat=repeat, frames=frames, interval=interval)

    def finalize(self, figure):
        plt.tight_layout()

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

    def save(self, figure, path: str, dpi: float):
        if isinstance(figure, plt.Figure):
            figure.savefig(path, dpi=dpi, transparent=True)
        elif isinstance(figure, animation.Animation):
            figure.save(path, dpi=dpi)
        else:
            raise ValueError(figure)


def _get_range(bounds: Box, index: int):
    lower = float(bounds.lower.vector[index].min)
    upper = float(bounds.upper.vector[index].max)
    return lower, upper


def _default_color(i: int):
    default_colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    return default_colors[i % len(default_colors)]


class LinePlot(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Grid) and data.spatial_rank == 1 and not instance(data)

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor):
        x = data.points.staggered_direction[0].vector[0].numpy()
        requires_legend = False
        if (color == None).all:
            color = math.range_tensor(channel(data))
        for c_idx, c_idx_n in zip(channel(data).meshgrid(), channel(data).meshgrid(names=True)):
            label = index_label(c_idx_n)
            values = data.values[c_idx].numpy()
            col = _rgba(color[c_idx])
            # color = _default_color(len(subplot.lines))
            if values.dtype in (np.complex64, np.complex128):
                subplot.plot(x, values.real, label=f"{label} real" if label else "real", color=col)
                subplot.plot(x, values.imag, '--', label=f"{label} imag" if label else "imag", color=col)
                requires_legend = True
            else:
                subplot.plot(x, values, label=label, color=col)
                requires_legend = requires_legend or label
        if requires_legend:
            subplot.legend()
        elif min_val is not None and max_val is not None:
            subplot.set_ylim((min_val - .02 * (max_val - min_val), max_val + .02 * (max_val - min_val)))
        return True


class Heatmap2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Grid) and channel(data).volume == 1 and data.spatial_rank == 2 and not instance(data)

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor):
        dims = spatial(data)
        vector = data.bounds.shape['vector']
        if data.bounds.upper.vector.item_names is not None:
            left, bottom = data.bounds.lower.vector[dims]
            right, top = data.bounds.upper.vector[dims]
        else:
            dim_indices = data.resolution.indices(dims)
            left, bottom = data.bounds.lower.vector[dim_indices]
            right, top = data.bounds.upper.vector[dim_indices]
        extent = (float(left), float(right), float(bottom), float(top))
        if space.spatial_rank == 3:  # surface plot
            z = data.values.numpy(dims)
            x, y = math.reshaped_numpy(data.points, [vector, *spatial(data)])
            im = subplot.plot_surface(x, y, z)
        else:  # heatmap
            aspect = subplot.get_aspect()
            im = subplot.imshow(data.values.numpy(dims.reversed), origin='lower', extent=extent, vmin=min_val, vmax=max_val, aspect=aspect)
        if show_color_bar:
            figure_has_color_bar = any(['colorbar' in ax.get_label() for ax in subplot.figure.axes])
            if min_val is None or max_val is None or not figure_has_color_bar:
                subplot.figure.colorbar(im, ax=subplot)  # adds a new Axis to the figure
        return True


class VectorField2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Grid) and data.spatial_rank == 2

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor):
        dims = space.vector.item_names
        vector = data.bounds.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        if isinstance(data, StaggeredGrid):
            data = data.at_centers()
        x, y = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
        u, v = math.reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.non_channel], force_expand=True)
        color = subplot.xaxis.label.get_color()
        for ch in range(u.shape[0]):
            subplot.quiver(x, y, u[ch], v[ch], color=color, units='xy', scale=1)
        return True


class VectorField3D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Grid) and channel(data).volume > 1 and data.spatial_rank == 3

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor):
        dims = space.vector.item_names
        vector = data.bounds.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        if isinstance(data, StaggeredGrid):
            data = data.at_centers()
        x, y, z = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
        u, v, w = math.reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.non_channel], force_expand=True)
        for ch in range(u.shape[0]):
            subplot.quiver(x, y, z, u[ch], v[ch], w[ch])


class Heatmap3D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Grid) and channel(data).volume == 1 and data.spatial_rank == 3

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor):
        dims = space.vector.item_names
        x, y, z = StaggeredGrid(lambda x: x, math.extrapolation.BOUNDARY, data.bounds, data.resolution).staggered_tensor().numpy(('vector',) + dims)
        values = data.values.numpy(dims)
        cmap = plt.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
        colors = cmap(norm(values))
        subplot.voxels(x, y, z, values, facecolors=colors, edgecolor='k')


class VectorCloud2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, PointCloud) and data.spatial_rank == 2 and 'vector' in channel(data)

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor):
        vector = data.points.shape['vector']
        x, y = math.reshaped_numpy(data.points, [vector, data.shape.without('vector')])
        u, v = math.reshaped_numpy(data.values, [vector, data.shape.without('vector')], force_expand=True)
        if color.shape:
            col = [_rgba(c) for c in color.numpy(data.shape.non_channel).reshape(-1)]
        else:
            col = _rgba(color)
        subplot.quiver(x, y, u, v, color=col, units='xy', scale=1)


class PointCloud2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, PointCloud) and data.spatial_rank == 2

    def plot(self, data: PointCloud, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor):
        dims = space.vector.item_names
        vector = data.bounds.shape['vector']
        channels = channel(data.points).without('vector')
        legend_patches = []
        if (color == None).all:
            color = math.range_tensor(channels)
        for idx, idx_n in zip(channels.meshgrid(), channels.meshgrid(names=True)):
            col = color[idx]
            PointCloud2D._plot_points(subplot, data[idx], dims, vector, col)
            if col.rank < color.rank:  # There are multiple colors
                legend_patches.append(Patch(color=_rgba(col), label=index_label(idx_n)))
        if legend_patches:
            subplot.legend(handles=legend_patches)

    @staticmethod
    def _plot_points(axis, data: PointCloud, dims, vector, color):
        if isinstance(data.elements, GeometryStack):
            for idx in data.elements.geometries.shape[0].meshgrid():
                PointCloud2D._plot_points(axis, data[idx], dims, vector, color[idx])
            return
        x, y = math.reshaped_numpy(data.points.vector[dims], [vector, non_channel(data)], force_expand=True)
        mpl_colors = matplotlib_colors(color, non_channel(data), default=0)
        if isinstance(data.elements, Point):
            if spatial(data.points).is_empty:
                axis.scatter(x, y, marker='x', color=mpl_colors, s=6 ** 2, alpha=0.8)
        else:
            if isinstance(data.elements, Sphere):
                rad = math.reshaped_numpy(data.elements.bounding_radius(), [data.shape.non_channel], force_expand=True)
                shapes = [plt.Circle((xi, yi), radius=ri, linewidth=0, alpha=0.8, facecolor=ci) for xi, yi, ri, ci in zip(x, y, rad, mpl_colors)]
            elif isinstance(data.elements, BaseBox):
                w2, h2 = math.reshaped_numpy(data.elements.bounding_half_extent(), ['vector', data.shape.non_channel], force_expand=True)
                shapes = [plt.Rectangle((xi - w2i, yi - h2i), w2i * 2, h2i * 2, linewidth=1, edgecolor='white', alpha=0.8, facecolor=ci) for xi, yi, w2i, h2i, ci in zip(x, y, w2, h2, mpl_colors)]
            else:
                rad = math.reshaped_numpy(data.elements.bounding_radius(), [data.shape.non_channel], force_expand=True)
                shapes = [plt.Circle((xi, yi), radius=ri, linewidth=0, alpha=0.8, facecolor=ci) for xi, yi, ri, ci in zip(x, y, rad, mpl_colors)]
            c = matplotlib.collections.PatchCollection(shapes, match_original=True)
            axis.add_collection(c)
        if spatial(data.points):  # Connect by line
            x, y = math.reshaped_numpy(data.points.vector[dims], [vector, instance(data), spatial(data)])
            mpl_colors = matplotlib_colors(color, instance(data))
            for i in range(instance(data).volume):
                marker = 'o' if isinstance(data.elements, Point) and spatial(data.elements).volume > 2 else None
                axis.plot(x[i], y[i], marker=marker, markersize=2.5, color=mpl_colors[i] if mpl_colors is not None else None)
        if any(non_channel(data).item_names):
            PointCloud2D._annotate_points(axis, data.points)

    @staticmethod
    def _annotate_points(axis, points: math.Tensor):
        labelled_dims = non_channel(points)
        labelled_dims = math.concat_shapes(*[d for d in labelled_dims if d.item_names[0]])
        if not labelled_dims:
            return
        if all(dim.name in points.shape.get_item_names('vector') for dim in labelled_dims):
            return  # The point labels match one of the figure axes, so they are redundant
        if points.shape['vector'].size == 2:
            xs, ys = math.reshaped_numpy(points, ['vector', points.shape.without('vector')], force_expand=True)
            labels = [index_label(idx) for idx in labelled_dims.meshgrid(names=True)]
            x_view = axis.get_xlim()[1] - axis.get_xlim()[0]
            y_view = axis.get_ylim()[1] - axis.get_ylim()[0]
            x_c = .95 * axis.get_xlim()[1] + .1 * axis.get_xlim()[0]
            y_c = .95 * axis.get_ylim()[1] + .1 * axis.get_ylim()[0]
            for x, y, label in zip(xs, ys, labels):
                if axis.get_xscale() == 'log':
                    offset_x = x * (1 + .0003 * x_view) if x < x_c else x * (1 - .0003 * x_view)
                else:
                    offset_x = x + .01 * x_view if x < x_c else x - .01 * x_view
                if axis.get_yscale() == 'log':
                    offset_y = y * (1 + .0003 * y_view) if y < y_c else y * (1 - .0003 * y_view)
                else:
                    offset_y = y + .01 * y_view if y < y_c else y - .01 * y_view
                axis.text(offset_x, offset_y, label)


class PointCloud3D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, PointCloud) and data.spatial_rank == 3

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor):
        dims = space.vector.item_names
        vector = data.bounds.shape['vector']
        channels = channel(data.points).without('vector')
        for idx in channels.meshgrid(names=True):
            x, y, z = math.reshaped_numpy(data[idx].points.vector[dims], [vector, non_channel(data)])
            mpl_colors = matplotlib_colors(color[idx], non_channel(data), default=0)
            M = subplot.transData.get_matrix()
            x_scale, y_scale, z_scale = M[0, 0], M[1, 1], M[2, 2]
            if isinstance(data.elements, Sphere):
                symbol = 'o'
                size = data.elements.bounding_radius().numpy() * 0.4
            elif isinstance(data.elements, BaseBox):
                symbol = 's'
                size = math.mean(data.elements.bounding_half_extent(), 'vector').numpy() * 0.35
            elif isinstance(data.elements, Point):
                symbol = 'x'
                size = 6 / (0.5 * (x_scale+y_scale+z_scale)/3)
            else:
                symbol = 'X'
                size = data.elements.bounding_radius().numpy()
            subplot.scatter(x, y, z, marker=symbol, color=mpl_colors, s=(size * 0.5 * (x_scale + y_scale + z_scale) / 3) ** 2)


def _rgba(col):
    if isinstance(col, Tensor):
        col = next(iter(col))
    if col is None:
        cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        return cycle[0]
    if not isinstance(col, (str, tuple, list)):
        cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        col = cycle[int(col) % len(cycle)]
    if isinstance(col, str) and col.startswith('#'):
        col = tuple(int(col.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    col = np.asarray(col)
    if col.dtype.kind == 'i':
        col = col / 255.
    return col


def matplotlib_colors(color: Tensor, dims: Shape, default=None) -> list or None:
    if color.rank == 0 and color.native() is None:
        if default is None:
            return None
        else:
            color = math.wrap(default)
    color = color[math.shape(color).without(dims).first_index()]  # Select first color along unlisted dimensions
    if color.dtype.kind == int:
        cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        return [cycle[int(color[idx])] for idx in dims.meshgrid()]
    else:
        return [color[idx].native() for idx in dims.meshgrid()]



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
            Heatmap2D(),
            VectorField2D(),
            VectorField3D(),
            Heatmap3D(),
            VectorCloud2D(),
            PointCloud2D(),
            PointCloud3D(),
        ])
