import matplotlib.pyplot as plt
import numpy as np

from phi import math
from . import Grid
from ._field import SampledField


def plot(field: SampledField, title=False, colorbar=False, cmap='magma', figsize=(12, 5), same_scale=True):
    if isinstance(field, Grid) and field.shape.channel.volume == 1:
        return _plot_scalar_grid(field, title=title, colorbar=colorbar, cmap=cmap, figsize=figsize, same_scale=same_scale)
    elif isinstance(field, Grid):
        return _plot_vector_field(field, title=title, colorbar=colorbar, cmap=cmap, figsize=figsize)
    raise NotImplementedError("Only scalar grids supported at this time.")
    # im = axes[i].imshow(velocity_grad.inflow_loc[i].values.vector[0].numpy('y,x'), origin='lower', cmap='bwr')


def _plot_scalar_grid(grid: Grid, title, colorbar, cmap, figsize, same_scale):
    batch_size = grid.shape.batch.volume
    values = math.join_dimensions(grid.values, grid.shape.channel, 'channel').channel[0]
    plt_args = {}
    if same_scale:
        plt_args['vmin'] = math.min(values).native()
        plt_args['vmax'] = math.max(values).native()
    b_values = math.join_dimensions(values, grid.shape.batch, 'batch')
    fig, axes = plt.subplots(1, batch_size, figsize=figsize)
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    for b in range(batch_size):
        im = axes[b].imshow(b_values.batch[b].numpy('y,x'), origin='lower', cmap=cmap, **plt_args)
        if title:
            if isinstance(title, str):
                sub_title = title
            elif title is True:
                sub_title = f"{b} of {grid.shape.batch}"
            elif isinstance(title, (tuple, list)):
                sub_title = title[b]
            else:
                sub_title = None
            if sub_title is not None:
                axes[b].set_title(sub_title)
        if colorbar:
            plt.colorbar(im, ax=axes[b])
    plt.tight_layout()
    return fig, axes


def _plot_vector_field(grid: Grid, title: bool, colorbar: bool, cmap: str, figsize):
    raise NotImplementedError()
