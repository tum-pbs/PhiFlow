import matplotlib.pyplot as plt
import numpy as np

from phi import math
from phi.geom import Box
from . import Grid, CenteredGrid, StaggeredGrid
from ._field import SampledField


def plot(field: SampledField, title=False, colorbar=False, cmap='magma', figsize=(12, 5), same_scale=True, **plt_args):
    batch_size = field.shape.batch.volume
    values = math.join_dimensions(field.values, field.shape.channel, 'channel').channel[0]
    fig, axes = plt.subplots(1, batch_size, figsize=figsize)
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    b_values = math.join_dimensions(values, field.shape.batch, 'batch')
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
    # Individual plots
    if isinstance(field, Grid) and field.shape.channel.volume == 1:
        if same_scale:
            plt_args['vmin'] = math.min(values).native()
            plt_args['vmax'] = math.max(values).native()
        for b in range(batch_size):
            im = axes[b].imshow(b_values.batch[b].numpy('y,x'), origin='lower', cmap=cmap, **plt_args)
            if colorbar:
                plt.colorbar(im, ax=axes[b])
    elif isinstance(field, Grid):
        if isinstance(field, StaggeredGrid):
            field = field.at_centers()
        for b in range(batch_size):
            x, y = field.points.vector.unstack_spatial('x,y', to_numpy=True)
            data = math.join_dimensions(field.values, field.shape.batch, 'batch').batch[b]
            u, v = data.vector.unstack_spatial('x,y', to_numpy=True)
            axes[b].quiver(x-u/2, y-v/2, u, v)
    else:
        raise NotImplementedError(f"No figure recipe for {field}")
    plt.tight_layout()
    return fig, axes
