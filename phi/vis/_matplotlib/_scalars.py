import os
import warnings
from numbers import Number
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy
import numpy as np

from phi import math
from phi.field import Scene
from phi.field._scene import _str
from phiml.math import Tensor, batch
from phiml.backend import ML_LOGGER
from phi.vis._plot_util import smooth_uniform_curve
from phi.vis._vis_base import display_name
from ._matplotlib_plots import MATPLOTLIB


def plot_scalars(scene: Union[str, tuple, list, Scene, math.Tensor],
                 names: Union[str, tuple, list, math.Tensor] = None,
                 reduce: Union[str, tuple, list, math.Shape] = 'names',
                 down='',
                 smooth=1,
                 smooth_alpha=0.2,
                 smooth_linewidth=2.,
                 size=(8, 6),
                 transform: Callable = None,
                 tight_layout=True,
                 grid: Union[str, dict] = 'y',
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
    warnings.warn("plot_scalars is deprecated. Use load_scalars() and plot() instead.", DeprecationWarning, stacklevel=2)
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
            ML_LOGGER.debug(f"Reading {os.path.join(path, f'log_{name}.txt')}")
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
                ML_LOGGER.debug(f"Reading {os.path.join(path, 'log_step_time.txt')}")
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
            ML_LOGGER.debug(f"Plotting curve {label}")
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
