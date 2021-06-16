import os
from numbers import Number
from typing import Callable

import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib import animation

from phi import math
from phi.geom import Sphere, BaseBox
from phi.math import channel, batch
from phi.vis._plot_util import smooth_uniform_curve
from phi.vis._vis_base import display_name
from phi.field import Grid, StaggeredGrid, PointCloud
from phi.field import Scene
from phi.field._field import SampledField
from phi.field._scene import _str


def plot(field: SampledField, title=False, show_color_bar=True, size=(12, 5), same_scale=True, **plt_args):
    """
    Creates a Matplotlib figure to display a single field or batch of fields.

    Use [`matplotlib.pyplot.show()`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.show.html) or
    [`matplotlib.pyplot.savefig()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html) to view the figure.

    Args:
        field: `SampledField`, may contain batch dimensions which will create subfigures.
        title: Figure title.
        show_color_bar: Whether to show a colorbar for heatmap plots.
        size: Figure (width, height) in inches.
        same_scale: Whether to use the same value scale for all subplots.
        **plt_args: Additional plotting arguments passed to Matplotlib.

    Returns:
        [Matplotlib figure](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure).
    """
    batch_size, b_values = _batch(field)
    fig, axes = plt.subplots(1, batch_size, figsize=size)
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    if title:
        for b in range(batch_size):
            if isinstance(title, str):
                sub_title = title
            elif title is True:
                sub_title = f"{b} of {field.shape.batch}"
            elif isinstance(title, (tuple, list)):
                sub_title = title[b]
            else:
                sub_title = None
            if sub_title is not None:
                axes[b].set_title(sub_title)
    _plot(field, b_values, axes, batch_size, show_color_bar, same_scale, **plt_args)
    plt.tight_layout()
    return fig


def animate(fields: SampledField, dim='frames',
            show_color_bar=False, size=(8, 6), same_scale=True, repeat=True, interval=200, **plt_args) -> animation.Animation:
    """
    Creates a Matplotlib animation from `fields`.
    `fields` may be a sequence of frames or a single `SampledField` instances with a `frames` dimension.

    Args:
        fields: `SampledField` with `frames` dimension or `tuple` or `list` of `SampledField`.
        show_color_bar: Whether to show a color bar
        size: Figure size
        same_scale: Whether to use the same scale, both temporally and for all sub-figures.
        repeat: Whether the video should loop.
        interval: Frame time in milliseconds.
        **plt_args: Further plotting arguments, see `plot()`.

    Returns:
        Matplotlib `Animation`
    """
    assert isinstance(fields, SampledField)
    assert dim in fields.shape, f"Animation dimension {dim} not present in data."
    fields = list(fields.unstack(dim))
    batch_size, b_values = _batch(fields[0])
    fig, axes = plt.subplots(1, batch_size, figsize=size)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    def func(frame: int):
        field = fields[frame]
        batch_size, b_values = _batch(field)
        for axis in axes:
            axis.clear()
        _plot(field, b_values, axes, batch_size, show_color_bar, same_scale, **plt_args)

    ani = animation.FuncAnimation(fig, func, init_func=lambda: axes, repeat=repeat, frames=len(fields), interval=interval)
    plt.close(fig)
    return ani


def _plot(field, b_values, axes, batch_size, show_color_bar, same_scale, **plt_args):
    if isinstance(field, Grid) and field.shape.channel.volume == 1:
        left, bottom = field.bounds.lower.vector.unstack_spatial('x,y')
        right, top = field.bounds.upper.vector.unstack_spatial('x,y')
        extent = (float(left), float(right), float(bottom), float(top))
        if same_scale:
            plt_args['vmin'] = math.min(b_values).native()
            plt_args['vmax'] = math.max(b_values).native()
        for b in range(batch_size):
            im = axes[b].imshow(b_values.batch[b].numpy('y,x'), origin='lower', extent=extent, **plt_args)
            if show_color_bar:
                plt.colorbar(im, ax=axes[b])
    elif isinstance(field, Grid):  # vector field
        if isinstance(field, StaggeredGrid):
            field = field.at_centers()
        for b in range(batch_size):
            x, y = [d.numpy('x,y') for d in field.points.vector.unstack_spatial('x,y')]
            data = math.join_dimensions(field.values, field.shape.batch, batch('batch')).batch[b]
            u, v = [d.numpy('x,y') for d in data.vector.unstack_spatial('x,y')]
            color = axes[b].xaxis.label.get_color()
            axes[b].quiver(x-u/2, y-v/2, u, v, color=color)
    elif isinstance(field, PointCloud):
        for b in range(batch_size):
            points = math.join_dimensions(field.points, field.points.shape.batch, batch('batch')).batch[b]
            x, y = [d.numpy() for d in points.vector.unstack_spatial('x,y')]
            color = [str(d) for d in field.color.points.unstack(len(x))]
            if field.bounds:
                lower_x, lower_y = [float(d) for d in field.bounds.lower.vector.unstack_spatial('x,y')]
                upper_x, upper_y = [float(d) for d in field.bounds.upper.vector.unstack_spatial('x,y')]
            else:
                lower_x, lower_y = [np.min(x), np.min(y)]
                upper_x, upper_y = [np.max(x), np.max(y)]
            if isinstance(field.elements, Sphere):
                shape = 'o'
                size = float(field.elements.bounding_radius())
            elif isinstance(field.elements, BaseBox):
                shape = 's'
                size = float(field.elements.bounding_half_extent())
            else:
                shape = 'X'
                size = float(field.elements.bounding_radius())
            axes[b].set_xlim((lower_x, upper_x))
            axes[b].set_ylim((lower_y, upper_y))
            M = axes[b].transData.get_matrix()
            x_scale, y_scale = M[0, 0], M[1, 1]
            axes[b].scatter(x, y, marker=shape, color=color, s=(size * 2 * x_scale) ** 2)
    else:
        raise NotImplementedError(f"No figure recipe for {field}")


def _batch(field: SampledField):
    if isinstance(field, PointCloud):
        batch_size = field.shape.batch.volume
    else:
        batch_size = field.shape.batch.volume
    values = math.join_dimensions(field.values, field.shape.channel, channel('channel')).channel[0]
    b_values = math.join_dimensions(values, field.shape.batch, batch('batch'))
    return batch_size, b_values


def plot_scalars(scene: str or tuple or list or Scene or math.Tensor,
                 names: str or tuple or list or math.Tensor = None,
                 reduce: str or tuple or list or math.Shape = 'names',
                 down='',
                 smooth=1,
                 smooth_alpha=0.2,
                 smooth_linewidth=1.,
                 size=(8, 6),
                 transform: Callable = None,
                 tight_layout=True,
                 grid='y',
                 log_scale='',
                 legend='upper right',
                 xlim=None,
                 ylim=None,
                 titles=True,
                 labels: math.Tensor = None,
                 xlabel: str = None,
                 ylabel: str = None,
                 colors: math.Tensor = 'default'):
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

    shape = (scene.shape & names.shape)
    batches = shape.without(reduce).without(additional_reduce)

    cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    fig, axes = plt.subplots(batches.only(down).volume, batches.without(down).volume, figsize=size)
    axes = axes if isinstance(axes, numpy.ndarray) else [axes]

    for b, axis in zip(batches.meshgrid(), axes):
        assert isinstance(axis, plt.Axes)
        names_equal = names[b].rank == 0
        paths_equal = scene.paths[b].rank == 0
        if titles:
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

        def single_plot(name, path, label, i, color):
            curve = numpy.loadtxt(os.path.join(path, f"log_{name}.txt"))
            if curve.ndim == 2:
                x, values, *_ = curve.T
            else:
                values = curve
                x = np.arange(len(values))
            if transform:
                x, values = transform(np.stack([x, values]))
            if color == 'default':
                color = cycle[i]
            elif isinstance(color, Number):
                color = cycle[int(color)]
            axis.plot(x, values, color=color, alpha=smooth_alpha, linewidth=1)
            axis.plot(*smooth_uniform_curve(x, values, n=smooth), color=color, linewidth=smooth_linewidth, label=label)
            if grid:
                grid_axis = 'both' if 'x' in grid and 'y' in grid else grid
                axis.grid(which='both', axis=grid_axis, linestyle='--')
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

        math.map(single_plot, names[b], scene.paths[b], curve_labels, math.range_tensor(shape.after_gather(b)), colors)
        if legend:
            axis.legend(loc=legend)
    # Final touches
    if tight_layout:
        plt.tight_layout()
    return fig


def savefig(filename: str, transparent=True):
    plt.savefig(filename, transparent=transparent)
