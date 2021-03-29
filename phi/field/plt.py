from collections import Callable

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from phi import math
from . import Grid, StaggeredGrid, PointCloud
from ._field import SampledField
from ._field_math import batch_stack


def plot(field: SampledField or tuple or list, title=False, colorbar=False, figsize=(12, 5), same_scale=True, **plt_args):
    if isinstance(field, (tuple, list)):
        field = batch_stack(*field, dim='fields')
    batch_size, b_values = _batch(field)
    fig, axes = plt.subplots(1, batch_size, figsize=figsize)
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
    _plot(field, b_values, axes, batch_size, colorbar, same_scale, **plt_args)
    plt.tight_layout()
    return fig, axes


def animate(fields: SampledField or tuple or list,
            colorbar=False, figsize=(8, 6), same_scale=True, repeat=True, interval=200, **plt_args) -> animation.Animation:
    """
    Creates a Matplotlib animation from `fields`.
    `fields` may be a sequence of frames or a single `SampledField` instances with a `frames` dimension.

    Args:
        fields: `SampledField` with `frames` dimension or `tuple` or `list` of `SampledField`.
        colorbar: Whether to show a color bar
        figsize: Figure size
        same_scale: Whether to use the same scale, both temporally and for all sub-figures.
        repeat: Whether the video should loop.
        interval: Frame time in milliseconds.
        **plt_args: Further plotting arguments, see `plot()`.

    Returns:
        Matplotlib `Animation`
    """
    if isinstance(fields, SampledField):
        assert 'frames' in fields.shape, "When passing a single Field, it must have a dimension with name 'frames'."
        fields = fields.unstack('frames')
    fields = list(fields)
    field = fields[0]
    batch_size, b_values = _batch(field)
    fig, axes = plt.subplots(1, batch_size, figsize=figsize)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    def func(frame: int):
        field = fields[frame]
        batch_size, b_values = _batch(field)
        for axis in axes:
            axis.clear()
        _plot(field, b_values, axes, batch_size, colorbar, same_scale, **plt_args)

    ani = animation.FuncAnimation(fig, func, init_func=lambda: axes, repeat=repeat, frames=len(fields), interval=interval)
    plt.close(fig)
    return ani


def _plot(field, b_values, axes, batch_size, colorbar, same_scale, **plt_args):
    if isinstance(field, Grid) and field.shape.channel.volume == 1:
        if same_scale:
            plt_args['vmin'] = math.min(b_values).native()
            plt_args['vmax'] = math.max(b_values).native()
        for b in range(batch_size):
            im = axes[b].imshow(b_values.batch[b].numpy('y,x'), origin='lower', **plt_args)
            if colorbar:
                plt.colorbar(im, ax=axes[b])
    elif isinstance(field, Grid):
        if isinstance(field, StaggeredGrid):
            field = field.at_centers()
        for b in range(batch_size):
            x, y = field.points.vector.unstack_spatial('x,y', to_numpy=True)
            data = math.join_dimensions(field.values, field.shape.batch, 'batch').batch[b]
            u, v = data.vector.unstack_spatial('x,y', to_numpy=True)
            color = axes[b].xaxis.label.get_color()
            axes[b].quiver(x-u/2, y-v/2, u, v, color=color)
    elif isinstance(field, PointCloud):
        for b in range(batch_size):
            points = math.join_dimensions(field.points, field.points.shape.batch.without('points'), 'batch').batch[b]
            x, y = points.vector.unstack_spatial('x,y', to_numpy=True)
            color = field.color.points.unstack(len(x), to_python=True)
            if field.bounds:
                lower = field.bounds.lower.vector.unstack_spatial('x,y', to_python=True)
                upper = field.bounds.upper.vector.unstack_spatial('x,y', to_python=True)
            else:
                lower = [np.min(x), np.min(y)]
                upper = [np.max(x), np.max(y)]
            axes[b].scatter(x, y, marker='o', color=color)
            axes[b].set_xlim((lower[0], upper[0]))
            axes[b].set_ylim((lower[1], upper[1]))
    else:
        raise NotImplementedError(f"No figure recipe for {field}")


def _batch(field: SampledField):
    if isinstance(field, PointCloud):
        batch_size = field.shape.without('points').batch.volume
    else:
        batch_size = field.shape.batch.volume
    values = math.join_dimensions(field.values, field.shape.channel, 'channel').channel[0]
    b_values = math.join_dimensions(values, field.shape.batch, 'batch')
    return batch_size, b_values
