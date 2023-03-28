import sys
import warnings
from typing import Callable, Tuple, Any, Dict, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import rc
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from phi import math, field
from phi.field import Grid, StaggeredGrid, PointCloud, SampledField
from phi.field._mesh import Mesh
from phi.geom import Sphere, BaseBox, Point, Box
from phi.geom._poly_surface import PolygonSurface
from phi.geom._stack import GeometryStack
from phi.math import Tensor, channel, spatial, instance, non_channel, Shape, reshaped_numpy, shape, non_instance
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
                        x, y = bounds.vector.item_names
                        axis.set_xlabel(display_name(x))
                        axis.set_ylabel(display_name(y))
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
                        # --- Remove labels if axes shared ---
                        for left_col in range(col):
                            if (row, left_col) in spaces and spaces[(row, left_col)].vector[y] == bounds.vector[y]:
                                axis.set_ylabel("")
                                axis.tick_params(labelleft=False)
                        for below_row in range(row + 1, rows + 1):
                            if (below_row, col) in spaces and spaces[(below_row, col)].vector[x] == bounds.vector[x]:
                                axis.set_xlabel("")
                                axis.tick_params(labelbottom=False)
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
        figure.tight_layout()
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
        pass

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

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        x = data.points.staggered_direction[0].vector[0].numpy()
        requires_legend = False
        if (color == None).all:
            color = math.range_tensor(channel(data))
        for c_idx, c_idx_n in zip(channel(data).meshgrid(), channel(data).meshgrid(names=True)):
            label = index_label(c_idx_n)
            values = data.values[c_idx].numpy()
            col = _plt_col(color[c_idx])
            alpha_f = float(alpha[c_idx])
            # color = _default_color(len(subplot.lines))
            if (err[c_idx] != 0).any:
                v_err = reshaped_numpy(err, [spatial(data)])
                subplot.fill_between(x, values - v_err, values + v_err, color=col, alpha=alpha_f * .2)
            if values.dtype in (np.complex64, np.complex128):
                subplot.plot(x, values.real, label=f"{label} real" if label else "real", color=col, alpha=alpha_f)
                subplot.plot(x, values.imag, '--', label=f"{label} imag" if label else "imag", color=col, alpha=alpha_f)
                requires_legend = True
            else:
                subplot.plot(x, values, label=label, color=col, alpha=alpha_f)
                requires_legend = requires_legend or label
        if requires_legend:
            subplot.legend()
        elif min_val is not None and max_val is not None:
            subplot.set_ylim((min_val - .02 * (max_val - min_val), max_val + .02 * (max_val - min_val)))


class BarChart(Recipe):

    def __init__(self, col_width=.8):
        self.col_width = col_width

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, PointCloud) and data.elements.vector.size == 1 and data.elements.vector.item_names == instance(data).names and spatial(data.values).is_empty

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        vector = data.bounds.shape['vector']
        x, = reshaped_numpy(data.points, [vector, instance(data)])
        x_range = x.max() - x.min()
        x_min = x.min() - x_range / (len(x)-1) / 2
        x_max = x.max() + x_range / (len(x)-1) / 2
        width = x_max - x_min
        channels = channel(data.values).volume
        for i, ch in enumerate(channel(data.values).meshgrid(names=True)):
            height = reshaped_numpy(data.values[ch], [instance(data)], force_expand=True)
            errs = reshaped_numpy(err[ch], [instance(data)], force_expand=True)
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
        subplot.set_xticks(x, instance(data).item_names[0])
        if channels > 1:
            subplot.legend()


class Heatmap2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Grid) and channel(data).volume == 1 and data.spatial_rank == 2 and not instance(data)

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
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
            x, y = reshaped_numpy(data.points, [vector, *spatial(data)])
            im = subplot.plot_surface(x, y, z)
        else:  # heatmap
            aspect = subplot.get_aspect()
            image = data.values.numpy(dims.reversed)
            if data.values.dtype.kind == complex:
                amplitude = abs(image) / np.max(abs(image))
                phase = np.angle(image) / (2*np.pi) + .5
                hsv = np.stack([phase, np.ones_like(amplitude), amplitude], -1)
                rgb = matplotlib.colors.hsv_to_rgb(hsv)
                im = subplot.imshow(rgb, origin='lower', extent=extent, vmin=min_val, vmax=max_val, aspect=aspect, alpha=float(alpha))
            else:
                im = subplot.imshow(image, origin='lower', extent=extent, vmin=min_val, vmax=max_val, aspect=aspect, alpha=float(alpha))
        if show_color_bar:
            figure_has_color_bar = any(['colorbar' in ax.get_label() for ax in subplot.figure.axes])
            if min_val is None or max_val is None or not figure_has_color_bar:
                if min_val is not None and max_val is not None:  # only one color bar for all subplots
                    figure.subplots_adjust(left=.1, bottom=.1, right=0.85, top=.92)
                    cax = figure.add_axes([0.87, 0.1, 0.08 if data.values.dtype.kind == complex else 0.03, 0.82])
                else:
                    cax = None
                if data.values.dtype.kind == complex:
                    if cax is not None:
                        amp, phase = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
                        cbar_img = matplotlib.colors.hsv_to_rgb(np.stack([phase, np.ones_like(phase), amp], -1))
                        cmap_extent = (min_val, max_val, 0, 2 * np.pi)
                        cax.imshow(cbar_img, extent=cmap_extent, aspect='auto', origin='lower')
                        cax.yaxis.tick_right()
                        cax.set_yticks([0, np.pi, 2 * np.pi], ['0', 'π', '2π'])
                        cax.yaxis.set_label_position('right')
                else:
                    subplot.figure.colorbar(im, ax=subplot if cax is None else None, cax=cax)  # adds a new Axis to the figure
        return True


class VectorField2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Grid) and data.spatial_rank == 2

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        vector = data.bounds.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        if isinstance(data, StaggeredGrid):
            data = data.at_centers()
        x, y = reshaped_numpy(data.points.vector[dims], [vector, data.shape.without('vector')])
        u, v = reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.without('vector')], force_expand=True)
        alphas = reshaped_numpy(alpha, [vector, data.shape.without('vector')], force_expand=True)
        colors = matplotlib_colors(color, data.shape.without('vector'))
        for ch in range(u.shape[0]):
            subplot.quiver(x, y, u[ch], v[ch], color=colors, alpha=alphas[ch], units='xy', scale=1)
        return True


class VectorField3D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Grid) and channel(data).volume > 1 and data.spatial_rank == 3

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        vector = data.bounds.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        if isinstance(data, StaggeredGrid):
            data = data.at_centers()
        x, y, z = reshaped_numpy(data.points.vector[dims], [vector, data.shape.without('vector')])
        u, v, w = reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.without('vector')], force_expand=True)
        alphas = reshaped_numpy(alpha, [vector, data.shape.without('vector')], force_expand=True)
        colors = matplotlib_colors(color, data.shape.without('vector'))
        for ch in range(u.shape[0]):
            subplot.quiver(x, y, z, u[ch], v[ch], w[ch], color=colors, alpha=alphas[ch])


class Heatmap3D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Grid) and channel(data).volume == 1 and data.spatial_rank == 3

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
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

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        vector = data.points.shape['vector']
        x, y = reshaped_numpy(data.points, [vector, data.shape.without('vector')])
        u, v = reshaped_numpy(data.values, [vector, data.shape.without('vector')], force_expand=True)
        if color.shape:
            col = [_plt_col(c) for c in color.numpy(data.shape.non_channel).reshape(-1)]
        else:
            col = _plt_col(color)
        alphas = reshaped_numpy(alpha, [data.shape.without('vector')], force_expand=True)
        subplot.quiver(x, y, u, v, color=col, units='xy', scale=1, alpha=alphas)


class PointCloud2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, PointCloud) and data.spatial_rank == 2

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        assert isinstance(data, PointCloud)
        dims = space.vector.item_names
        vector = data.bounds.shape['vector']
        channels = channel(data.points).without('vector')
        legend_patches = []
        if (color == None).all:
            color = math.range_tensor(channels)
        for idx, idx_n in zip(channels.meshgrid(), channels.meshgrid(names=True)):
            col = color[idx]
            PointCloud2D._plot_points(subplot, data[idx], dims, vector, col, alpha[idx], err[idx])
            if col.rank < color.rank:  # There are multiple colors
                legend_patches.append(Patch(color=_plt_col(col), label=index_label(idx_n)))
        if legend_patches:
            subplot.legend(handles=legend_patches)

    @staticmethod
    def _plot_points(axis: Axes, data: PointCloud, dims: tuple, vector: Shape, color: Tensor, alpha: Tensor, err: Tensor):
        if isinstance(data.elements, GeometryStack):
            for idx in data.elements.geometries.shape[0].meshgrid():
                PointCloud2D._plot_points(axis, data[idx], dims, vector, color[idx], alpha[idx], err[idx])
            return
        x, y = reshaped_numpy(data.points.vector[dims], [vector, non_channel(data)], force_expand=True)
        mpl_colors = matplotlib_colors(color, non_channel(data), default=0)
        alphas = reshaped_numpy(alpha, [non_channel(data)], force_expand=True)
        if isinstance(data.elements, Point):
            if spatial(data.points).is_empty:
                axis.scatter(x, y, marker='x', color=mpl_colors, s=6 ** 2, alpha=alphas)
                if (err != 0).any:
                    x_err = reshaped_numpy(err.vector[dims[0]], [instance(data)]) if dims[0] in err.vector.item_names else 0
                    y_err = reshaped_numpy(err.vector[dims[1]], [instance(data)]) if dims[1] in err.vector.item_names else 0
                    axis.errorbar(x, y, y_err, x_err, fmt=' ')
        else:
            if isinstance(data.elements, Sphere):
                rad = reshaped_numpy(data.elements.bounding_radius(), [data.shape.non_channel], force_expand=True)
                shapes = [plt.Circle((xi, yi), radius=ri, linewidth=0, alpha=a, facecolor=ci) for xi, yi, ri, ci, a in zip(x, y, rad, mpl_colors, alphas)]
            elif isinstance(data.elements, BaseBox):
                w2, h2 = reshaped_numpy(data.elements.bounding_half_extent(), ['vector', data.shape.non_channel], force_expand=True)
                shapes = [plt.Rectangle((xi - w2i, yi - h2i), w2i * 2, h2i * 2, linewidth=1, edgecolor='white', alpha=a, facecolor=ci) for xi, yi, w2i, h2i, ci, a in zip(x, y, w2, h2, mpl_colors, alphas)]
            elif isinstance(data.elements, PolygonSurface):
                xs, ys = reshaped_numpy(data.elements.corners(), ['vector', data.shape.non_channel, 'vertex_index'], force_expand=True)
                counts = reshaped_numpy(data.elements._vertex_count, [data.shape.non_channel], force_expand=True)
                shapes = [plt.Polygon(np.stack([x[:count], y[:count]], -1), closed=True, edgecolor='white', alpha=a, facecolor=ci) for x, y, count, ci, a in zip(xs, ys, counts, mpl_colors, alphas)]
            else:
                rad = reshaped_numpy(data.elements.bounding_radius(), [data.shape.non_channel], force_expand=True)
                shapes = [plt.Circle((xi, yi), radius=ri, linewidth=0, alpha=a, facecolor=ci) for xi, yi, ri, ci, a in zip(x, y, rad, mpl_colors, alphas)]
            c = matplotlib.collections.PatchCollection(shapes, match_original=True)
            axis.add_collection(c)
        if spatial(data.points):  # Connect by line
            for idx in instance(data).meshgrid():
                x, y = reshaped_numpy(data[idx].points.vector[dims], [vector, spatial(data)])
                col = _plt_col(color)
                alpha_f = float(alpha[idx].max)
                if (err[idx] != 0).any:
                    x_err = reshaped_numpy(err[idx].vector[dims[0]], [spatial(data)]) if dims[0] in err.vector.item_names else 0
                    y_err = reshaped_numpy(err[idx].vector[dims[1]], [spatial(data)]) if dims[1] in err.vector.item_names else 0
                    if math.max(x_err) > math.max(y_err):
                        axis.fill_betweenx(y, x - x_err, x + x_err, color=col, alpha=alpha_f * .2)
                    else:
                        axis.fill_between(x, y - y_err, y + y_err, color=col, alpha=alpha_f * .2)
                axis.plot(x, y, color=col, alpha=alpha_f)
                if isinstance(data.elements, Point) and 2 < spatial(data.elements).volume < 100:
                    axis.scatter(x, y, s=3, marker='o', c=col, alpha=alphas)

        if any(non_channel(data).item_names):
            PointCloud2D._annotate_points(axis, data.points, color, alpha)

    @staticmethod
    def _annotate_points(axis, points: math.Tensor, color: Tensor, alpha: Tensor):
        labelled_dims = non_channel(points)
        labelled_dims = math.concat_shapes(*[d for d in labelled_dims if d.item_names[0]])
        if not labelled_dims:
            return
        if all(dim.name in points.shape.get_item_names('vector') for dim in labelled_dims):
            return  # The point labels match one of the figure axes, so they are redundant
        if points.shape['vector'].size == 2:
            xs, ys = reshaped_numpy(points, ['vector', points.shape.without('vector')], force_expand=True)
            x_view = axis.get_xlim()[1] - axis.get_xlim()[0]
            y_view = axis.get_ylim()[1] - axis.get_ylim()[0]
            x_c = .95 * axis.get_xlim()[1] + .1 * axis.get_xlim()[0]
            y_c = .95 * axis.get_ylim()[1] + .1 * axis.get_ylim()[0]
            for x, y, idx in zip(xs, ys, labelled_dims.meshgrid(names=True)):
                if axis.get_xscale() == 'log':
                    offset_x = x * (1 + .0003 * x_view) if x < x_c else x * (1 - .0003 * x_view)
                else:
                    offset_x = x + .01 * x_view if x < x_c else x - .01 * x_view
                if axis.get_yscale() == 'log':
                    offset_y = y * (1 + .0003 * y_view) if y < y_c else y * (1 - .0003 * y_view)
                else:
                    offset_y = y + .01 * y_view if y < y_c else y - .01 * y_view
                axis.text(offset_x, offset_y, index_label(idx), color=_plt_col(color[idx]), alpha=float(alpha[idx]))


class PointCloud3D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, PointCloud) and data.spatial_rank == 3

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        vector = data.bounds.shape['vector']
        channels = channel(data.points).without('vector')
        for idx in channels.meshgrid(names=True):
            x, y, z = reshaped_numpy(data[idx].points.vector[dims], [vector, non_channel(data)])
            mpl_colors = matplotlib_colors(color[idx], non_channel(data), default=0)
            alphas = reshaped_numpy(alpha[idx], [non_channel(data)], force_expand=True)
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
            subplot.scatter(x, y, z, marker=symbol, color=mpl_colors, alpha=alphas, s=(size * 0.5 * (x_scale + y_scale + z_scale) / 3) ** 2)


class Mesh2D(Recipe):

    def can_plot(self, data: SampledField, space: Box) -> bool:
        return isinstance(data, Mesh) and data.spatial_rank == 2

    def plot(self, data: SampledField, figure, subplot, space: Box, min_val: float, max_val: float, show_color_bar: bool, color: Tensor, alpha: Tensor, err: Tensor):
        dims = space.vector.item_names
        assert isinstance(data, Mesh)
        point_cloud = PointCloud(data.elements, data.values, data.extrapolation, bounds=data.bounds)
        PointCloud2D().plot(point_cloud, figure, subplot, space, min_val, max_val, show_color_bar, color, alpha, err)
        i, j = math.nonzero(data.edges).vector
        i_x, i_y = reshaped_numpy(data.points[{instance(data).name: i}][dims], ['vector', 'nonzero'], force_expand=True)
        j_x, j_y = reshaped_numpy(data.points[{instance(data).name: j}][dims], ['vector', 'nonzero'], force_expand=True)
        subplot.plot(np.stack([i_x, j_x]), np.stack([i_y, j_y]), color=_plt_col(color), alpha=float(alpha.max))


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
    if color.rank == 0 and color.native() is None:
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
            BarChart(),
            Mesh2D(),
        ])
