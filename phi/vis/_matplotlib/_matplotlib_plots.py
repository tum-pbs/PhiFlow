import logging
import os
from numbers import Number
from typing import Callable, Tuple, Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib import animation

from phi import math, field
from phi.geom import Sphere, BaseBox
from phi.math import Tensor, batch, channel, instance, spatial
from phi.vis._plot_util import smooth_uniform_curve
from phi.vis._vis_base import display_name, PlottingLibrary
from phi.field import Grid, StaggeredGrid, PointCloud, Scene, unstack, SampledField
from phi.field._scene import _str


class MatplotlibPlots(PlottingLibrary):

    def __init__(self):
        super().__init__('matplotlib', [plt.Figure])

    def create_figure(self,
                      size: tuple,
                      rows: int,
                      cols: int,
                      subplots: Dict[Tuple[int, int], int],
                      titles: Tensor) -> Tuple[Any, Dict[Tuple[int, int], Any]]:
        figure, axes = plt.subplots(rows, cols, figsize=size)
        self.current_figure = figure
        axes = np.reshape(axes, (rows, cols))
        axes_by_pos = {}
        for row in range(rows):
            for col in range(cols):
                axes[row, col].set_title(titles.rows[row].cols[col].native())
                if (row, col) not in subplots:
                    axes[row, col].remove()
                else:
                    if subplots[(row, col)] == 3:
                        axes[row, col].remove()
                        axes[row, col] = figure.add_subplot(rows, cols, cols*row + col + 1, projection='3d')
                    axes_by_pos[(row, col)] = axes[row, col]
        return figure, axes_by_pos

    def plot(self,
             data: SampledField,
             figure,
             subplot,
             min_val: float = None,
             max_val: float = None,
             show_color_bar: bool = True,
             **plt_args):
        """
        Returns:
            [Matplotlib figure](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure).
        """
        plt.tight_layout()
        _plot(subplot, data, show_color_bar=show_color_bar, vmin=min_val, vmax=max_val, **plt_args)
        plt.tight_layout()
        return figure

    def show(self, figure: plt.Figure):
        figure.show()

    def save(self, figure: plt.Figure, path: str, dpi: float):
        figure.savefig(path, dpi=dpi)


MATPLOTLIB = MatplotlibPlots()


# def animate(fields: SampledField,
#             dim='frames',
#             repeat=True,
#             interval=200,
#             title=False,
#             size=(8, 6),
#             show_color_bar=False,
#             same_scale=True,
#             **plt_args) -> animation.Animation:
#     """
#     Creates a Matplotlib animation from `fields`.
#     `fields` may be a sequence of frames or a single `SampledField` instances with a `frames` dimension.
#
#     Args:
#         fields: `SampledField` with `frames` dimension or `tuple` or `list` of `SampledField`.
#         dim: Time dimension to animate (default=`'frames'`).
#         repeat: Whether the video should loop.
#         interval: Frame time in milliseconds.
#         title: Figure/sub-figure title. If `str` or `tuple`/`list` of `str`. `True` to generate a title automatically.
#         size: Figure size
#         show_color_bar: Whether to show a color bar
#         same_scale: Whether to use the same scale, both temporally and for all sub-figures.
#         **plt_args: Further plotting arguments, see `plot()`.
#
#     Returns:
#         Matplotlib `Animation`
#     """
#     assert isinstance(fields, SampledField)
#     assert dim in fields.shape, f"Animation dimension {dim} not present in data."
#     fields = list(fields.unstack(dim))
#     fig, _ = _subplots(fields[0], size, None)
#
#     def func(frame: int):
#         field = fields[frame]
#         for axis in fig.axes:
#             axis.clear()
#         MATPLOTLIB.plot(field, fig, )
#         plot(field, existing_figure=fig, title=title, show_color_bar=show_color_bar, same_scale=same_scale, **plt_args)
#
#     ani = animation.FuncAnimation(fig, func, init_func=lambda: fig.axes, repeat=repeat, frames=len(fields), interval=interval)
#     plt.close(fig)
#     return ani


def _plot(axis, data, show_color_bar, vmin, vmax, **plt_args):
    if isinstance(data, Grid) and data.spatial_rank == 1:
        x = data.points.staggered_direction[0].vector[0].numpy()
        for c in channel(data).meshgrid():
            values = data.values[c].numpy()
            if values.dtype in (np.complex64, np.complex128):
                axis.plot(x, values.real, label='real')
                axis.plot(x, values.imag, label='imag')
                axis.legend()
            else:
                axis.plot(x, values)
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
        im = axis.imshow(data.values.numpy(dims.reversed), origin='lower', extent=extent, vmin=vmin, vmax=vmax, **plt_args)
        if show_color_bar:
            plt.colorbar(im, ax=axis)
        axis.set_xlabel(dims.names[0])
        axis.set_ylabel(dims.names[1])
    elif isinstance(data, Grid) and data.spatial_rank == 2:  # vector field
        if isinstance(data, StaggeredGrid):
            data = data.at_centers()
        x, y = [d.numpy('x,y') for d in data.points.vector.unstack_spatial('x,y')]
        u, v = [d.numpy('x,y') for d in data.values.vector.unstack_spatial('x,y')]
        color = axis.xaxis.label.get_color()
        axis.quiver(x, y, u, v, color=color, units='xy', scale=1)
        axis.set_aspect('equal', adjustable='box')
    elif isinstance(data, Grid) and channel(data).volume > 1 and data.spatial_rank == 3:
        x, y, z = [d.numpy('x,y,z') for d in data.points.vector.unstack_spatial('x,y,z')]
        u, v, w = [d.numpy('x,y,z') for d in data.values.vector.unstack_spatial('x,y,z')]
        axis.quiver(x, y, z, u, v, w)
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_zlabel('z')
    elif isinstance(data, Grid) and channel(data).volume == 1 and data.spatial_rank == 3:
        x, y, z = [d.numpy('x,y,z') for d in data.points.vector.unstack_spatial('x,y,z')]
        values = data.values.numpy('x,y,z')
        cmap = plt.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
        colors = cmap(norm(values))
        axis.voxels(values, facecolors=colors, edgecolor='k')
    elif isinstance(data, PointCloud) and data.spatial_rank == 2:
        axis.set_aspect('equal', adjustable='box')
        lower_x, lower_y = [float(d) for d in data.bounds.lower.vector.unstack_spatial('x,y')]
        upper_x, upper_y = [float(d) for d in data.bounds.upper.vector.unstack_spatial('x,y')]
        axis.set_xlim((lower_x, upper_x))
        axis.set_ylim((lower_y, upper_y))
        if data.points.shape.non_channel.rank > 1:
            data_list = field.unstack(data, data.points.shape.non_channel[0].name)
            for d in data_list:
                _plot_points(axis, d, **plt_args)
        else:
            _plot_points(axis, data, **plt_args)
    elif isinstance(data, PointCloud) and data.spatial_rank == 3:
        if data.points.shape.non_channel.rank > 1:
            data_list = field.unstack(data, data.points.shape.non_channel[0].name)
            for d in data_list:
                _plot(axis, d, show_color_bar, vmin, vmax, **plt_args)
        else:
            x, y, z = [d.numpy() for d in data.points.vector.unstack_spatial('x,y,z')]
            color = [str(d) for d in data.color.points.unstack(len(x))]
            if isinstance(data.elements, Sphere):
                symbol = 'o'
                size = data.elements.bounding_radius().numpy() * 0.4
            elif isinstance(data.elements, BaseBox):
                symbol = 's'
                size = math.mean(data.elements.bounding_half_extent(), 'vector').numpy() * 0.35
            else:
                symbol = 'X'
                size = data.elements.bounding_radius().numpy()
            M = axis.transData.get_matrix()
            x_scale, y_scale, z_scale = M[0, 0], M[1, 1], M[2, 2]
            axis.scatter(x, y, z, marker=symbol, color=color, s=(size * 0.5 * (x_scale+y_scale+z_scale)/3) ** 2)
        lower_x, lower_y, lower_z = [float(d) for d in data.bounds.lower.vector.unstack_spatial('x,y,z')]
        upper_x, upper_y, upper_z = [float(d) for d in data.bounds.upper.vector.unstack_spatial('x,y,z')]
        axis.set_xlim((lower_x, upper_x))
        axis.set_ylim((lower_y, upper_y))
        axis.set_zlim((lower_z, upper_z))
    else:
        raise NotImplementedError(f"No figure recipe for {data}")


def _plot_points(axis, data, **plt_args):
    x, y = [d.numpy() for d in data.points.vector.unstack_spatial('x,y')]
    color = [str(d) for d in data.color.points.unstack(len(x))]
    if isinstance(data.elements, Sphere):
        symbol = 'o'
        size = data.elements.bounding_radius().numpy() * 1.41
    elif isinstance(data.elements, BaseBox):
        symbol = 's'
        size = math.mean(data.elements.bounding_half_extent(), 'vector').numpy()
    else:
        symbol = 'X'
        size = data.elements.bounding_radius().numpy()
    size_px = size * _get_pixels_per_unit(axis.figure, axis)
    axis.scatter(x, y, marker=symbol, color=color, s=size_px ** 2, alpha=0.8)


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
                 colors: math.Tensor = 'default'):
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
    if not isinstance(colors, math.Tensor):
        colors = math.wrap(colors)
    if xlabel is None:
        xlabel = 'Iterations' if x == 'steps' else 'Time (s)'

    shape = (scene.shape & names.shape)
    batches = shape.without(reduce).without(additional_reduce)

    cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    fig, axes = plt.subplots(batches.only(down).volume, batches.without(down).volume, figsize=size)
    axes = axes if isinstance(axes, numpy.ndarray) else [axes]

    for b, axis in zip(batches.meshgrid(), axes):
        assert isinstance(axis, plt.Axes)
        names_equal = names[b].rank == 0
        paths_equal = scene.paths[b].rank == 0
        if titles is not None and titles is not False:
            if isinstance(titles, str):
                axis.set_title(titles)
            elif names_equal:
                axis.set_title(display_name(str(names[b])))
            elif paths_equal:
                axis.set_title(os.path.basename(str(scene.paths[b])))
        if labels is not None:
            curve_labels = labels
        elif names_equal:
            curve_labels = math.map(os.path.basename, scene.paths[b])
        elif paths_equal:
            curve_labels = names[b]
        else:
            curve_labels = math.map(lambda p, n: f"{os.path.basename(p)} - {n}", scene.paths[b], names[b])

        def single_plot(name, path, label, i, color, smooth):
            logging.debug(f"Reading {os.path.join(path, f'log_{name}.txt')}")
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
                logging.debug(f"Reading {os.path.join(path, 'log_step_time.txt')}")
                _, x_values, *_ = numpy.loadtxt(os.path.join(path, "log_step_time.txt")).T
                values = values[:len(x_values)]
                x_values = np.cumsum(x_values[:len(values)])
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
            logging.debug(f"Plotting curve {label}")
            axis.plot(x_values, values, color=color, alpha=smooth_alpha, linewidth=1)
            curve = np.stack([x_values, values], -1)
            axis.plot(*smooth_uniform_curve(curve, smooth), color=color, linewidth=smooth_linewidth, label=label)
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

        math.map(single_plot, names[b], scene.paths[b], curve_labels, math.range_tensor(shape.after_gather(b)), colors, smooth)
        if legend:
            axis.legend(loc=legend)
    # Final touches
    if tight_layout:
        plt.tight_layout()
    return fig


def savefig(filename: str, transparent=True):
    plt.savefig(filename, transparent=transparent)
