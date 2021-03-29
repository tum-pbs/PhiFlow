from typing import Callable

import numpy
import matplotlib.pyplot as plt
import os

from phi import math
from phi.app._app import display_name
from phi.field import Scene
from phi.field._scene import _str


# def smooth_curve(x, y):
# from math import erf
#     erf(hi) - erf(lo)


def smooth_uniform_curve(array, n=16):
    if n == 1:
        return numpy.arange(len(array)), array
    if len(array) <= n:
        mean = numpy.tile(numpy.mean(array, -1, keepdims=True), 2)
        return numpy.arange(2), mean
    arrays = [array[i:i-n+1 or None] for i in range(n)]
    result = numpy.mean(arrays, axis=0)
    return numpy.arange(n//2-1, len(array)-n//2), result


def plot_scalars(scene: str or tuple or list or Scene or math.Tensor,
                 names: str or tuple or list or math.Tensor = None,
                 reduce: str or tuple or list or math.Shape = 'names',
                 smooth=1,
                 smooth_alpha=0.4,
                 figsize=(8, 6),
                 transform: Callable = None,
                 tight_layout=True):
    scene = Scene.at(scene)
    additional_reduce = ()
    if names is None:
        first_path = next(iter(math.flatten(scene.paths)))
        names = [_str(n) for n in os.listdir(first_path)]
        names = [n[4:-4] for n in names if n.endswith('.txt') and n.startswith('log_')]
        names = math.wrap(names, 'names')
        additional_reduce = ['names']
    elif isinstance(names, str):
        names = math.wrap(names)
    elif isinstance(names, (tuple, list)):
        names = math.wrap(names, 'names')
    else:
        assert isinstance(names, math.Tensor), f"Invalid argument 'names': {type(names)}"

    shape = (scene.shape & names.shape)
    batch = shape.without(reduce).without(additional_reduce)

    cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    fig, axes = plt.subplots(1, batch.volume, figsize=figsize)
    axes = axes if isinstance(axes, numpy.ndarray) else [axes]

    for b, axis in zip(batch.meshgrid(), axes):
        assert isinstance(axis, plt.Axes)
        names_equal = names[b].rank == 0
        paths_equal = scene.paths[b].rank == 0
        if names_equal:
            axis.set_title(display_name(str(names[b])))
        elif paths_equal:
            axis.set_title(os.path.basename(str(scene.paths[b])))

        def single_plot(name, path, i):
            curve = numpy.loadtxt(os.path.join(path, f"log_{name}.txt"))
            name = display_name(name)
            if transform:
                curve = transform(curve)
            if names_equal:
                label = os.path.basename(path)
            elif paths_equal:
                label = name
            else:
                label = f"{os.path.basename(path)} - {name}"
            axis.plot(curve, color=cycle[i], alpha=smooth_alpha, linewidth=1)
            axis.plot(*smooth_uniform_curve(curve, n=smooth), color=cycle[i], linewidth=2, label=label)
            return name

        math.map(single_plot, names[b], scene.paths[b], math.range_tensor(shape.after_gather(b)))
        axis.legend()
    # Final touches
    if tight_layout:
        plt.tight_layout()
    return fig
