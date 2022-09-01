import logging
import os
import sys
from numbers import Number
from typing import Callable, Tuple, Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib import animation, cbook
from matplotlib import rc
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D

from phi import math, field
from phi.field import Grid, StaggeredGrid, PointCloud, Scene, SampledField
from phi.field._scene import _str
from phi.geom import Sphere, BaseBox, Point, Box
from phi.math import Tensor, batch, channel, spatial, instance, non_channel
from phi.math.backend import PHI_LOGGER
from phi.vis._plot_util import smooth_uniform_curve
from phi.vis._vis_base import display_name, PlottingLibrary


class MatplotlibPlots(PlottingLibrary):

    def __init__(self):
        super().__init__('matplotlib', [plt.Figure, animation.Animation])

    def create_figure(self,
                      size: tuple,
                      rows: int,
                      cols: int,
                      spaces: Dict[Tuple[int, int], Box],
                      titles: Dict[Tuple[int, int], str]) -> Tuple[Any, Dict[Tuple[int, int], Any]]:
        figure, axes = plt.subplots(rows, cols, figsize=size)
        self.current_figure = figure
        axes = np.reshape(axes, (rows, cols))
        axes_by_pos = {}
        for row in range(rows):
            for col in range(cols):
                axis = axes[row, col]
                if (row, col) not in spaces:
                    axis.remove()
                else:
                    bounds = spaces[(row, col)]
                    if bounds.spatial_rank == 1:
                        axis.set_xlabel(bounds.vector.item_names[0])
                        axis.set_xlim(_get_range(bounds, 0))
                    elif bounds.spatial_rank == 2:
                        axis.set_xlabel(bounds.vector.item_names[0])
                        axis.set_ylabel(bounds.vector.item_names[1])
                        axis.set_xlim(_get_range(bounds, 0))
                        axis.set_ylim(_get_range(bounds, 1))
                        axis.set_aspect('equal', adjustable='box')
                    elif bounds.spatial_rank == 3:
                        axis.remove()
                        axis = axes[row, col] = figure.add_subplot(rows, cols, cols*row + col + 1, projection='3d')
                        axis.set_xlabel(bounds.vector.item_names[0])
                        axis.set_ylabel(bounds.vector.item_names[1])
                        axis.set_zlabel(bounds.vector.item_names[2])
                        axis.set_xlim(_get_range(bounds, 0))
                        axis.set_ylim(_get_range(bounds, 1))
                        axis.set_zlim(_get_range(bounds, 2))
                    axis.set_title(titles.get((row, col), None))
                    axes_by_pos[(row, col)] = axes[row, col]
        return figure, axes_by_pos

    def animate(self, fig: plt.Figure, frames: int, plot_frame_function: Callable, interval: float, repeat: bool):
        if 'ipykernel' in sys.modules:
            rc('animation', html='html5')

        base_axes = tuple(fig.axes)
        positions = {a: (a.figbox.p0, a.figbox.p1) for a in base_axes}
        titles = {a: a.get_title() for a in base_axes}
        specs = {a: a.get_subplotspec() for a in base_axes}

        def clear_and_plot(frame: int):
            axes = tuple(fig.axes)
            for axis in axes:
                if axis not in base_axes:  # colorbar etc.
                    axis.remove()
                else:
                    # axis.cla()  # this also clears titles and axis labels
                    axis.lines.clear()
                    axis.patches.clear()
                    axis.texts.clear()
                    axis.tables.clear()
                    axis.artists.clear()
                    axis.images.clear()
                    axis.collections.clear()

                    box = Bbox(positions[axis])
                    axis.set_position(box, which='active')
                    axis.set_subplotspec(specs[axis])
                    # axis.set_title(titles[axis])
            # plt.tight_layout()
            plot_frame_function(frame)

        return animation.FuncAnimation(fig, clear_and_plot, repeat=repeat, frames=frames, interval=interval)

    def plot(self,
             data: SampledField,
             figure,
             subplot,
             space: Box,
             min_val: float = None,
             max_val: float = None,
             show_color_bar: bool = True,
             **plt_args):
        """
        Returns:
            [Matplotlib figure](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure).
        """
        # plt.tight_layout()
        _plot(subplot, data, space, show_color_bar=show_color_bar, vmin=min_val, vmax=max_val, **plt_args)
        plt.tight_layout()
        return figure

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

    def save(self, figure, path: str, dpi: float):
        if isinstance(figure, plt.Figure):
            figure.savefig(path, dpi=dpi, transparent=True)
        elif isinstance(figure, animation.Animation):
            figure.save(path, dpi=dpi)
        else:
            raise ValueError(figure)


MATPLOTLIB = MatplotlibPlots()


def _get_range(bounds: Box, index: int):
    lower = float(bounds.lower.vector[index].min)
    upper = float(bounds.upper.vector[index].max)
    return lower, upper


def _default_color(i: int):
    default_colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    return default_colors[i % len(default_colors)]


def _plot(axis, data: SampledField, space: Box, show_color_bar, vmin, vmax, **plt_args):
    dims = space.vector.item_names
    # dims = data.bounds.vector.item_names
    vector = data.bounds.shape['vector']
    extra_channels = data.shape.channel.without('vector')
    if isinstance(data, Grid) and data.spatial_rank == 1:  # Line plot
        x = data.points.staggered_direction[0].vector[0].numpy()
        requires_legend = False
        for c in channel(data).meshgrid(names=True):
            label = ", ".join([i for dim, i in c.items() if isinstance(i, str)])
            values = data.values[c].numpy()
            color = _default_color(len(axis.lines))
            if values.dtype in (np.complex64, np.complex128):
                axis.plot(x, values.real, label=f"real({label})" if label else "real", color=color)
                axis.plot(x, values.imag, '--', label=f"imag({label})" if label else "imag", color=color)
                requires_legend = True
            else:
                axis.plot(x, values, label=label, color=color)
                requires_legend = requires_legend or label
        if requires_legend:
            axis.legend()
        if data.values.dtype.kind != complex and data.values.min > 0 and data.values.max > 100 * data.values.min:
            axis.set_yscale('log')
        elif vmin is not None and vmax is not None:
            axis.set_ylim((vmin - .02 * (vmax - vmin), vmax + .02 * (vmax - vmin)))
    elif isinstance(data, Grid) and channel(data).volume == 1 and data.spatial_rank == 2:
        dims = spatial(data)
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
            im = axis.plot_surface(x, y, z, **plt_args)
        else:  # heatmap
            im = axis.imshow(data.values.numpy(dims.reversed), origin='lower', extent=extent, vmin=vmin, vmax=vmax, **plt_args)
        if show_color_bar:
            figure_has_color_bar = any(['colorbar' in ax.get_label() for ax in axis.figure.axes])
            if vmin is None or vmax is None or not figure_has_color_bar:
                axis.figure.colorbar(im, ax=axis)  # adds a new Axis to the figure
    elif isinstance(data, Grid) and data.spatial_rank == 2:  # vector field
        if isinstance(data, StaggeredGrid):
            data = data.at_centers()
        x, y = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
        u, v = math.reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.non_channel], force_expand=True)
        color = axis.xaxis.label.get_color()
        for ch in range(u.shape[0]):
            axis.quiver(x, y, u[ch], v[ch], color=color, units='xy', scale=1)
    elif isinstance(data, Grid) and channel(data).volume > 1 and data.spatial_rank == 3:  # 3D vector field
        if isinstance(data, StaggeredGrid):
            data = data.at_centers()
        x, y, z = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
        u, v, w = math.reshaped_numpy(data.values.vector[dims], [vector, extra_channels, data.shape.non_channel], force_expand=True)
        for ch in range(u.shape[0]):
            axis.quiver(x, y, z, u[ch], v[ch], w[ch])
    elif isinstance(data, Grid) and channel(data).volume == 1 and data.spatial_rank == 3:  # 3D heatmap
        x, y, z = StaggeredGrid(lambda x: x, math.extrapolation.BOUNDARY, data.bounds, data.resolution).staggered_tensor().numpy(('vector',) + dims)
        values = data.values.numpy(dims)
        cmap = plt.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
        colors = cmap(norm(values))
        axis.voxels(x, y, z, values, facecolors=colors, edgecolor='k')
    elif isinstance(data, PointCloud) and data.spatial_rank == 2 and 'vector' in channel(data):
        axis.set_aspect('equal', adjustable='box')
        vector = data.points.shape['vector']
        x, y = math.reshaped_numpy(data.points, [vector, data.shape.without('vector')])
        u, v = math.reshaped_numpy(data.values, [vector, data.shape.without('vector')], force_expand=True)
        if data.color.shape:
            color = data.color.numpy(data.shape.non_channel).reshape(-1)
        else:
            color = data.color.native()
        axis.quiver(x, y, u, v, color=color, units='xy', scale=1)
    elif isinstance(data, PointCloud) and data.spatial_rank == 2:
        axis.set_aspect('equal', adjustable='box')
        if data.points.shape.without('vector').rank > 1:  # multiple instance / spatial dimensions
            data_list = field.unstack(data, data.points.shape.without('vector')[0].name)
            for d in data_list:
                _plot_points(axis, d, dims, vector, **plt_args)
        else:
            _plot_points(axis, data, dims, vector, **plt_args)
    elif isinstance(data, PointCloud) and data.spatial_rank == 3:
        if data.points.shape.non_channel.rank > 1:
            data_list = field.unstack(data, data.points.shape.non_channel[0].name)
            for d in data_list:
                _plot(axis, d, show_color_bar, vmin, vmax, **plt_args)
        else:
            x, y, z = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
            color = [d.native() for d in data.color.points.unstack(len(x))]
            M = axis.transData.get_matrix()
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
            axis.scatter(x, y, z, marker=symbol, color=color, s=(size * 0.5 * (x_scale+y_scale+z_scale)/3) ** 2)
    else:
        raise NotImplementedError(f"No figure recipe for {data}")


def _plot_points(axis, data: PointCloud, dims, vector, **plt_args):
    x, y = math.reshaped_numpy(data.points.vector[dims], [vector, data.shape.non_channel])
    color = [d.native() for d in data.color.points.unstack(len(x))]
    if isinstance(data.elements, Point):
        axis.scatter(x, y, marker='x', color=color, s=6 ** 2, alpha=0.8)
    else:
        if isinstance(data.elements, Sphere):
            rad = math.reshaped_numpy(data.elements.bounding_radius(), [data.shape.non_channel], force_expand=True)
            shapes = [plt.Circle((xi, yi), radius=ri, linewidth=0, alpha=0.8, facecolor=ci) for xi, yi, ri, ci in zip(x, y, rad, color)]
        elif isinstance(data.elements, BaseBox):
            w2, h2 = math.reshaped_numpy(data.elements.bounding_half_extent(), ['vector', data.shape.non_channel], force_expand=True)
            shapes = [plt.Rectangle((xi-w2i, yi-h2i), w2i*2, h2i*2, linewidth=1, edgecolor='white', alpha=0.8, facecolor=ci) for xi, yi, w2i, h2i, ci in zip(x, y, w2, h2, color)]
        else:
            rad = math.reshaped_numpy(data.elements.bounding_radius(), [data.shape.non_channel], force_expand=True)
            shapes = [plt.Circle((xi, yi), radius=ri, linewidth=0, alpha=0.8, facecolor=ci) for xi, yi, ri, ci in zip(x, y, rad, color)]
        c = matplotlib.collections.PatchCollection(shapes, match_original=True)
        axis.add_collection(c)
    if non_channel(data).rank == 1 and non_channel(data).item_names[0]:
        _annotate_points(axis, data.points, non_channel(data))


def _annotate_points(axis, points: math.Tensor, labelled_dim: math.Shape):
    if points.shape['vector'].size == 2:
        x, y = math.reshaped_native(points, ['vector', points.shape.without('vector')], to_numpy=True, force_expand=True)
        if labelled_dim.item_names[0]:
            x_view = axis.get_xlim()[1] - axis.get_xlim()[0]
            y_view = axis.get_ylim()[1] - axis.get_ylim()[0]
            for x_, y_, label in zip(x, y, labelled_dim.item_names[0]):
                axis.annotate(label, (x_ + .01 * x_view, y_ + .01 * y_view))



def _get_pixels_per_unit(fig: plt.Figure, axis: plt.Axes, dpi=90):
    M = axis.transData.get_matrix()
    x_scale, y_scale = M[0, 0], M[1, 1]  # fig_size_px/unit
    # subplot_width = axis.figbox.width * axis.figure.bbox_inches.width
    # subplot_height = axis.figbox.height * axis.figure.bbox_inches.height
    # units_x = axis.get_xlim()[1] - axis.get_xlim()[0]
    # units_y = axis.get_ylim()[1] - axis.get_ylim()[0]
    # result_x = subplot_width * dpi / units_x
    # result_y = subplot_height * dpi / units_y
    return min(x_scale, y_scale)



def plot_scalars(scene: str or tuple or list or Scene or math.Tensor,
                 names: str or tuple or list or math.Tensor = None,
                 reduce: str or tuple or list or math.Shape = 'names',
                 down='',
                 smooth=1,
                 smooth_alpha=0.2,
                 smooth_linewidth=2.,
                 size=(8, 6),
                 transform: Callable = None,
                 tight_layout=True,
                 grid: str or dict = 'y',
                 log_scale='',
                 legend='upper right',
                 x='steps',
                 xlim=None,
                 ylim=None,
                 titles=True,
                 labels: math.Tensor = None,
                 xlabel: str = None,
                 ylabel: str = None,
                 colors: math.Tensor = 'default',
                 dashed: math.Tensor = False):
    """

    Args:
        scene: `str` or `Tensor`. Scene paths containing the data to plot.
        names: Data files to plot for each scene. The file must be located inside the scene directory and have the name `log_<name>.txt`.
        reduce: Tensor dimensions along which all curves are plotted in the same diagram.
        down: Tensor dimensions along which diagrams are ordered top-to-bottom instead of left-to-right.
        smooth: `int` or `Tensor`. Number of data points to average, -1 for all.
        smooth_alpha: Opacity of the non-smoothed curves under the smoothed curves.
        smooth_linewidth: Line width of the smoothed curves.
        size: Figure size in inches.
        transform: Function `T(x,y) -> (x,y)` transforming the curves.
        tight_layout:
        grid:
        log_scale:
        legend:
        x:
        xlim:
        ylim:
        titles:
        labels:
        xlabel:
        ylabel:
        colors: Line colors as `str`, `int` or `Tensor`. Integers are interpreted as indices of the default color list.

    Returns:
        MatPlotLib [figure](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure)
    """
    scene = Scene.at(scene)
    additional_reduce = ()
    if names is None:
        first_path = next(iter(math.flatten(scene.paths)))
        names = [_str(n) for n in os.listdir(first_path)]
        names = [n[4:-4] for n in names if n.endswith('.txt') and n.startswith('log_')]
        names = math.wrap(names, batch('names'))
        additional_reduce = ['names']
    elif isinstance(names, str):
        names = math.wrap(names)
    elif isinstance(names, (tuple, list)):
        names = math.wrap(names, batch('names'))
    else:
        assert isinstance(names, math.Tensor), f"Invalid argument 'names': {type(names)}"
    colors = math.wrap(colors)
    dashed = math.wrap(dashed)
    if xlabel is None:
        xlabel = 'Iterations' if x == 'steps' else 'Time (s)'

    shape = (scene.shape & names.shape)
    batches = shape.without(reduce).without(additional_reduce)

    cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    fig, axes = plt.subplots(batches.only(down).volume, batches.without(down).volume, figsize=size)
    MATPLOTLIB.current_figure = fig
    axes = axes if isinstance(axes, numpy.ndarray) else np.array(axes)

    for b, axis in zip(math.concat_shapes(batches.only(down), batches.without(down)).meshgrid(), axes.flatten()):
        assert isinstance(axis, plt.Axes)
        names_equal = names[b].rank == 0
        paths_equal = scene.paths[b].rank == 0
        if titles is not None and titles is not False:
            if isinstance(titles, str):
                axis.set_title(titles)
            elif isinstance(titles, Tensor):
                axis.set_title(titles[b].native())
            elif names_equal:
                axis.set_title(display_name(names[b].native()))
            elif paths_equal:
                axis.set_title(os.path.basename(scene.paths[b].native()))
        if labels is not None:
            curve_labels = labels
        elif names_equal:
            curve_labels = math.map(os.path.basename, scene.paths[b])
        elif paths_equal:
            curve_labels = names[b]
        else:
            curve_labels = math.map(lambda p, n: f"{os.path.basename(p)} - {n}", scene.paths[b], names[b])

        def single_plot(name, path, label, i, color, dashed_, smooth):
            PHI_LOGGER.debug(f"Reading {os.path.join(path, f'log_{name}.txt')}")
            curve = numpy.loadtxt(os.path.join(path, f"log_{name}.txt"))
            if curve.ndim == 2:
                x_values, values, *_ = curve.T
            else:
                values = curve
                x_values = np.arange(len(values))
            if x == 'steps':
                pass
            else:
                assert x == 'time', f"x must be 'steps' or 'time' but got {x}"
                PHI_LOGGER.debug(f"Reading {os.path.join(path, 'log_step_time.txt')}")
                _, x_values, *_ = numpy.loadtxt(os.path.join(path, "log_step_time.txt")).T
                values = values[:len(x_values+1)]
                x_values = np.cumsum(x_values[:len(values)-1])
                x_values = np.concatenate([[0.], x_values])
            if transform:
                x_values, values = transform(np.stack([x_values, values]))
            if color == 'default':
                color = cycle[i]
            try:
                color = int(color)
            except ValueError:
                pass
            if isinstance(color, Number):
                color = cycle[int(color)]
            PHI_LOGGER.debug(f"Plotting curve {label}")
            if smooth > 1:
                axis.plot(x_values, values, color=color, alpha=smooth_alpha, linewidth=1)
                curve = np.stack([x_values, values], -1)
                axis.plot(*smooth_uniform_curve(curve, smooth).T, *(['--'] if dashed_ else []), color=color, linewidth=smooth_linewidth, label=label)
            else:
                axis.plot(x_values, values, *(['--'] if dashed_ else []), color=color, linewidth=1, label=label)
            if grid:
                if isinstance(grid, dict):
                    axis.grid(**grid)
                else:
                    grid_axis = 'both' if 'x' in grid and 'y' in grid else grid
                    axis.grid(which='both', axis=grid_axis, linestyle='--', linewidth=size[1] * 0.3)
            if 'x' in log_scale:
                axis.set_xscale('log')
            if 'y' in log_scale:
                axis.set_yscale('log')
            if xlim:
                axis.set_xlim(xlim)
            if ylim:
                axis.set_ylim(ylim)
            if xlabel:
                axis.set_xlabel(xlabel)
            if ylabel:
                axis.set_ylabel(ylabel)
            return name

        math.map(single_plot, names[b], scene.paths[b], curve_labels, math.range_tensor(shape.after_gather(b)), colors, dashed, smooth)
        if legend:
            axis.legend(loc=legend)
    # Final touches
    if tight_layout:
        plt.tight_layout()
    return fig


def savefig(filename: str, transparent=True):
    plt.savefig(filename, transparent=transparent)
