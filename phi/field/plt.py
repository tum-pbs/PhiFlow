import matplotlib.pyplot as plt
import numpy as np

from phi import math
from . import Grid
from ._field import SampledField


def plot(field: SampledField, title=False, colorbar=False, cmap='magma'):
    if isinstance(field, Grid) and field.shape.channel.volume == 1:
        return _plot_scalar_grid(field, title=title, colorbar=colorbar, cmap=cmap)
    elif isinstance(field, Grid):
        return _plot_vector_field(field, title=title, colorbar=colorbar, cmap=cmap)
    raise NotImplementedError("Only scalar grids supported at this time.")
    # im = axes[i].imshow(velocity_grad.inflow_loc[i].values.vector[0].numpy('y,x'), origin='lower', cmap='bwr')


def _plot_scalar_grid(grid: Grid, title: bool, colorbar: bool, cmap: str):
    batch_size = grid.shape.batch.volume
    values = math.join_dimensions(grid.values, grid.shape.channel, 'channel').channel[0]
    b_values = math.join_dimensions(values, grid.shape.batch, 'batch')
    fig, axes = plt.subplots(1, batch_size, figsize=(16, 5))
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    for b in range(batch_size):
        im = axes[b].imshow(b_values.batch[b].numpy('y,x'), origin='lower', cmap=cmap)
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


def _plot_vector_field(grid: Grid, title: bool, colorbar: bool, cmap: str):
    raise NotImplementedError()
