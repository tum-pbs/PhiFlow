import inspect
import os
import sys
import warnings
from contextlib import contextmanager
from threading import Thread
from typing import Tuple, List, Dict, Union, Sequence, Any

from phiml.math._magic_ops import tree_map
from ._user_namespace import get_user_namespace, UserNamespace, DictNamespace
from ._viewer import create_viewer, Viewer
from ._vis_base import Control, value_range, Action, VisModel, Gui, PlottingLibrary, common_index, to_field, \
    get_default_limits, uniform_bound, is_jupyter, requires_color_map, display_name
from ._vis_base import title_label
from .. import math
from ..field import Scene, Field, PointCloud
from ..field._scene import _slugify_filename
from ..geom import Geometry, Box, embed
from phiml.math import Tensor, layout, batch, Shape, concat, vec, wrap, stack
from phiml.math._shape import parse_dim_order, DimFilter, EMPTY_SHAPE, merge_shapes, shape, channel
from phiml.math._tensors import Layout


def show(*fields: Union[Field, Tensor, Geometry, list, tuple, dict],
         lib: Union[str, PlottingLibrary] = None,
         row_dims: DimFilter = None,
         col_dims: DimFilter = batch,
         animate: DimFilter = None,
         overlay: DimFilter = 'overlay',
         title: Union[str, Tensor, list, tuple] = None,
         size=None,  # (12, 5),
         same_scale: Union[bool, Shape, tuple, list, str] = True,
         log_dims: Union[str, tuple, list, Shape] = '',
         show_color_bar=True,
         color: Union[str, int, Tensor, list, tuple] = None,
         alpha: Union[float, Tensor, list, tuple] = 1.,
         err: Union[Tensor, tuple, list, float] = 0.,
         frame_time=100,
         repeat=True,
         plt_params: Dict = None,
         max_subfigures=20):
    """
    Args:
        See `plot()`.
    """
    if not fields:  # only show, no plot
        if lib is not None:
            plots = get_plots(lib)
        else:
            if not LAST_FIGURE:
                warnings.warn("No plot yet created with phi.vis; nothing to show.", RuntimeWarning)
                return
            plots = get_plots_by_figure(LAST_FIGURE[0])
        return plots.show(plots.current_figure)
    else:
        kwargs = locals()
        del kwargs['fields']
        fig = plot(*fields, **kwargs)
        plots = get_plots_by_figure(fig)
        if isinstance(fig, Tensor):
            for fig in fig:
                plots.show(fig)
        else:
            return plots.show(fig)


def show_hist(data: Tensor, bins=math.instance(bins=20), weights=1, same_bins: DimFilter = None):
    hist, edges, center = math.histogram(data, bins, weights, same_bins)
    show(PointCloud(center, hist))


def close(figure=None):
    """
    Close and destroy a figure.

    Args:
        figure: (Optional) A figure that was created using `plot()`.
            If not specified, closes the figure created most recently.
    """
    if figure is None:
        figure = LAST_FIGURE[0]
    if isinstance(figure, Tensor):
        for fig in figure:
            close(fig)
    else:
        plots = get_plots_by_figure(figure)
        plots.close(figure)


close_ = close


def _default_field_variables(user_namespace: UserNamespace, fields: tuple):
    names = []
    values = []
    if len(fields) == 0:  # view all Fields
        user_variables = user_namespace.list_variables(only_public=True, only_current_scope=True)
        for name, val in user_variables.items():
            if isinstance(val, Field):
                names.append(name)
                values.append(val)
    else:  # find variable names
        user_variables = user_namespace.list_variables()
        for field in fields:
            if isinstance(field, str):
                split = [n.strip() for n in field.split(',')]
                names.extend(split)
                values.extend([user_namespace.get_variable(n, default=None) for n in split])
            else:
                for name, val in user_variables.items():
                    if val is field:
                        names.append(name)
                        values.append(field)
    return {n: v for n, v in zip(names, values)}


CONTROL_VARS = {}


def control(value, range: tuple = None, description="", **kwargs):
    """
    Mark a variable as controllable by any GUI created via `view()`.

    Example:
    >>> dt = control(1.0, (0.1, 10), name="Time increment (dt)")

    This will cause a control component (slider, checkbox, text field, drop-down, etc.) to be generated in the user interface.
    Changes to that component will immediately be reflected in the Python variable assigned to the control.
    The Python variable will always hold a primitive type, such as `int`, `float´, `bool` or `str`.

    Args:
        value: Initial value. Must be either `int`, `float`, `bool` or `str`.
        range: (Optional) Specify range of possible values as `(min, max)`. Only for `int`, `float` and `str` values.
        description: Human-readable description.
        **kwargs: Additional arguments to determine the appearance of the GUI component,
            e.g. `rows` for text fields or `log=False` for float sliders.

    Returns:
        `value`
    """
    assert type(value) in (int, float, bool, str), f"Value must be one of (int, float, bool, str) but {type(value)}"
    calling_code = inspect.stack()[1].code_context[0]
    assert 'control' in calling_code and '=' in calling_code, f"control() must be used in a variable assignment statement but context is: {calling_code}"
    calling_code = calling_code[:calling_code.index('control')]
    var_names = [var.strip() for var in calling_code.split('=')[:-1]]
    var_names = [n for n in var_names if n]
    for var_name in var_names:
        ctrl = Control(var_name, type(value), value, range, description, kwargs)
        value_range(ctrl)  # checks if valid
        CONTROL_VARS[var_name] = ctrl
    return value


ACTIONS = {}


def action(fun):
    doc = inspect.getdoc(fun)
    ACTIONS[Action(fun.__name__, doc)] = fun
    return fun


LAST_FIGURE = [None]  # reference to last figure (1-element list)


def get_current_figure():
    """
    Returns the figure that was most recently created using `plot()`.

    The type of the figure depends on which library was used, e.g. `matplotlib.figure.Figure` or `plotly.graph_objs.Figure`.
    """
    return LAST_FIGURE[0]


def plot(*fields: Union[Field, Tensor, Geometry, list, tuple, dict],
         lib: Union[str, PlottingLibrary] = None,
         row_dims: DimFilter = None,
         col_dims: DimFilter = batch,
         animate: DimFilter = None,
         overlay: DimFilter = 'overlay',
         title: Union[str, Tensor, list, tuple] = None,
         size=None,  # (12, 5),
         same_scale: Union[bool, Shape, tuple, list, str] = True,
         log_dims: Union[str, tuple, list, Shape] = '',
         show_color_bar=True,
         color: Union[str, int, Tensor, list, tuple] = None,
         alpha: Union[float, Tensor, list, tuple] = 1.,
         err: Union[Tensor, tuple, list, float] = 0.,
         frame_time=100,
         repeat=True,
         plt_params: Dict = None,
         max_subfigures=20):
    """
    Creates one or multiple figures and sub-figures and plots the given fields.

    To show the figures, use `show()`.

    The arguments `row_dims`, `col_dims`, `animate` and `overlay` control how data is presented.
    Each accepts dimensions as a `str`, `Shape`, tuple, list or type function.
    In addition to the dimensions present on the data to be plotted, the dimensions `args` is created if multiple arguments are passed,
    and `tuple`, `list`, `dict` are generated for corresponding objects to be plotted.

    Args:
        fields: Fields or Tensors to plot.
        lib: Plotting library name or reference. Valid names are `'matplotlib'`, `'plotly'` and `'console'`.
        row_dims: Batch dimensions along which sub-figures should be laid out vertically.
            `Shape` or comma-separated names as `str`, `tuple` or `list`.
        col_dims: Batch dimensions along which sub-figures should be laid out horizontally.
            `Shape` or comma-separated names as `str`, `tuple` or `list`.
        title: `str` for figures with a single subplot.
            For subplots, pass a string `Tensor` matching the content dimensions, i.e. `row_dims` and `col_dims`.
            Passing a `tuple`, `list` or `dict`, will create a tensor with these names internally.
        size: Figure size in inches, `(width, height)`.
        same_scale: Whether to use the same axis limits for all sub-figures.
        log_dims: Dimensions for which the plot axes should be scaled logarithmically.
            Can be given as a comma-separated `str`, a sequence of dimension names or a `Shape`.
            Use `'_'` to scale unnamed axes logarithmically, e.g. the y-axis of scalar functions.
        show_color_bar: Whether to display color bars for heat maps.
        color: `Tensor` of line / marker colors.
            The color can be specified either as a cycle index (int tensor) or as a hex code (str tensor).
            The color of different lines and markers can vary.
        alpha: Opacity as `float` or `Tensor`.
            This affects all elements, not only line plots.
            Opacity can vary between lines and markers.
        err: Expected deviation from the value given in `fields`.
            For supported plots, adds error bars of size *2·err*.
            If the plotted data is the mean of some distribution, a good choice for `err` is the standard deviation along the mean dims.
        animate: Time dimension to animate.
            If not present in the data, will produce a regular plot instead.
        overlay: Dimensions along which elements should be overlaid in the same subplot.
            The default is only the `overlay` dimension which is created by `overlay()`.
        frame_time: Interval between frames in the animation.
        repeat: Whether the animation should loop.

    Returns:
        `Tensor` of figure objects.
        The tensor contains those dimensions of `fields` that were not reduced by `row_dims`, `col_dims` or `animate`.
        Currently, only single-figure plots are supported.

        In case of an animation, a displayable animation object will be returned instead of a `Tensor`.
    """
    data = layout([layout_pytree_node(f) for f in fields], batch('args'))
    overlay = data.shape.only(overlay)
    animate = data.shape.only(animate).without(overlay)
    row_dims: Shape = data.shape.only(row_dims).without(animate).without(overlay)
    col_dims = data.shape.only(col_dims).without(row_dims).without(animate).without(overlay)
    fig_shape = batch(data).without(row_dims).without(col_dims).without(animate).without(overlay)
    reduced_shape = row_dims & col_dims & animate & overlay
    nrows = uniform_bound(row_dims).volume
    ncols = uniform_bound(col_dims).volume
    assert nrows * ncols <= max_subfigures, f"Too many subfigures ({nrows * ncols}) for max_subfigures={max_subfigures}. If you want to plot this many subfigures, increase the limit."
    positioning, indices = layout_sub_figures(data, row_dims, col_dims, animate, overlay, 0, 0)
    # --- Process arguments ---
    plots = default_plots(positioning) if lib is None else get_plots(lib)
    plt_params = {} if plt_params is None else dict(**plt_params)
    size = (None, None) if size is None else size
    if title is None:
        title_by_subplot = {pos: title_label(common_index(*i, exclude=reduced_shape.singleton)) for pos, i in indices.items()}
    elif isinstance(title, Tensor) and ('rows' in title.shape or 'cols' in title.shape):
        title_by_subplot = {(row, col): title.rows[row].cols[col].native() for (row, col) in positioning}
    else:
        title = layout_pytree_node(title, wrap_leaf=True)
        title_by_subplot = {pos: _title(title, i[0]) for pos, i in indices.items()}
    log_dims = parse_dim_order(log_dims) or ()
    color = layout_pytree_node(color, wrap_leaf=True)
    color = layout_color(positioning, indices, color)
    alpha = layout_pytree_node(alpha, wrap_leaf=True)
    alpha = tree_map(lambda x: 1 if x is None else x, alpha)
    err = layout_pytree_node(err, wrap_leaf=True)
    if same_scale is True:
        same_scale = '_'
    elif same_scale is False or same_scale is None:
        same_scale = ''
    same_scale = parse_dim_order(same_scale)
    if '_' in same_scale:
        if any([f.values.dtype.kind == complex for l in positioning.values() for f in l]):
            min_val = 0
            max_val = max([float(abs(f.values).finite_max) for l in positioning.values() for f in l] or [0])
        else:
            fin_min = lambda t: float(math.map(lambda f: math.finite_min(f.values, shape), t, dims=object).finite_min)
            fin_max = lambda t: float(math.map(lambda f: math.finite_max(f.values, shape), t, dims=object).finite_max)
            min_val = min([fin_min(f) for l in positioning.values() for f in l] or [0])
            max_val = max([fin_max(f) for l in positioning.values() for f in l] or [0])
            if min_val != min_val:  # NaN
                min_val = None
            if max_val != max_val:  # NaN
                max_val = None
    else:
        min_val = max_val = None
    # --- Layout ---
    subplots = {pos: _space(*fields, ignore_dims=animate, log_dims=log_dims, errs=[err[i] for i in indices[pos]]) for pos, fields in positioning.items()}
    subplots = {pos: _insert_value_dim(space, pos, subplots, min_val, max_val) for pos, space in subplots.items()}
    if same_scale:
        shared_lim: Box = share_axes(*subplots.values(), axes=same_scale)
        subplots = {pos: replace_bounds(lim, shared_lim) for pos, lim in subplots.items()}
    # --- animate or plot ---
    figures = []
    for plot_idx in fig_shape.meshgrid():
        figure, axes = plots.create_figure(size, nrows, ncols, subplots, log_dims, plt_params)
        if animate:
            def plot_frame(figure, frame: int):
                for pos, fields in positioning.items():
                    plots.set_title(title_by_subplot[pos], figure, axes[pos])
                    plots.set_title(display_name(animate.item_names[0][frame]) if animate.item_names[0] else None, figure, None)
                    for i, f in enumerate(fields):
                        idx = indices[pos][i]
                        f = f[{animate.name: int(frame)}]
                        plots.plot(f, figure, axes[pos], subplots[pos], min_val, max_val, show_color_bar, color[pos][i], alpha[idx], err[idx])
                plots.finalize(figure)
            anim = plots.animate(figure, animate.size, plot_frame, frame_time, repeat, interactive=True, time_axis=animate.name)
            if is_jupyter():
                plots.close(figure)
            LAST_FIGURE[0] = anim
            if fig_shape.volume == 1:
                return anim
            figures.append(anim)
        else:  # non-animated plot
            for pos, fields in positioning.items():
                plots.set_title(title_by_subplot[pos], figure, axes[pos])
                for i, f in enumerate(fields):
                    idx = indices[pos][i]
                    plots.plot(f, figure, axes[pos], subplots[pos], min_val, max_val, show_color_bar, color[pos][i], alpha[idx], err[idx])
            plots.finalize(figure)
            LAST_FIGURE[0] = figure
            figures.append(figure)
    return stack([layout(f) for f in figures], fig_shape) if fig_shape else figures[0]

def layout_pytree_node(data, wrap_leaf=False):
    # we could wrap instead of layout if all values have same shapes
    if isinstance(data, tuple):
        return layout(tuple([layout_pytree_node(i) for i in data]), batch('tuple'))
    elif isinstance(data, list):
        return layout([layout_pytree_node(i) for i in data], batch('list'))
    elif isinstance(data, dict):
        return layout({k: layout_pytree_node(v) for k, v in data.items()}, batch('dict'))
    return wrap(data) if wrap_leaf else data


def layout_sub_figures(data: Union[Tensor, Field],
                       row_dims: Shape,
                       col_dims: Shape,
                       animate: Shape,  # do not reduce these dims, has priority
                       overlay: Shape,
                       offset_row: int,
                       offset_col: int,
                       positioning: Dict[Tuple[int, int], List] = None,
                       indices: Dict[Tuple[int, int], List[dict]] = None,
                       base_index: Dict[str, Union[int, str]] = None) -> Tuple[Dict[Tuple[int, int], List[Field]], Dict[Tuple[int, int], List[dict]]]:
    if positioning is None:
        assert indices is None and base_index is None
        positioning = {}
        indices = {}
        base_index = {}
    # --- if data is a group of elements, lay them out recursively ---
    if isinstance(data, Tensor) and data.dtype.kind == object:  # layout
        if not data.shape:  # nothing to plot
            return positioning, indices
        dim0 = data.shape[0]
        if dim0.only(overlay):
            for overlay_index in dim0.only(overlay).meshgrid(names=True):  # overlay these fields
                # ToDo expand constants along rows/cols
                layout_sub_figures(data[overlay_index], row_dims, col_dims, animate, overlay, offset_row, offset_col, positioning, indices, {**base_index, **overlay_index})
            return positioning, indices
        elif dim0.only(animate):
            pass
        else:
            elements = math.unstack(data, dim0.name)
            offset = 0
            for item_name, e in zip(dim0.get_item_names(dim0.name) or range(dim0.size), elements):
                index = dict(base_index, **{dim0.name: item_name})
                if dim0.only(row_dims):
                    layout_sub_figures(e, row_dims, col_dims, animate, overlay, offset_row + offset, offset_col, positioning, indices, index)
                    offset += shape(e).only(row_dims).volume
                elif dim0.only(col_dims):
                    layout_sub_figures(e, row_dims, col_dims, animate, overlay, offset_row, offset_col + offset, positioning, indices, index)
                    offset += shape(e).only(col_dims).volume
                else:
                    layout_sub_figures(e, row_dims, col_dims, animate, overlay, offset_row, offset_col, positioning, indices, index)
            return positioning, indices
    # --- data must be a plottable object ---
    data = to_field(data)
    overlay = data.shape.only(overlay)
    animate = data.shape.only(animate).without(overlay)
    row_shape = data.shape.only(row_dims).without(animate).without(overlay)
    col_shape = data.shape.only(col_dims).without(row_dims).without(animate).without(overlay)
    row_shape &= row_dims.after_gather(base_index)
    col_shape &= col_dims.after_gather(base_index)
    for ri, r in enumerate(row_shape.meshgrid(names=True)):
        for ci, c in enumerate(col_shape.meshgrid(names=True)):
            for o in overlay.meshgrid(names=True):
                sub_data = data[r][c][o]
                positioning.setdefault((offset_row + ri, offset_col + ci), []).append(sub_data)
                indices.setdefault((offset_row + ri, offset_col + ci), []).append(dict(base_index, **r, **c, **o))
    return positioning, indices


def _space(*values: Field or Tensor, ignore_dims: Shape, log_dims: Tuple[str], errs: Sequence[Tensor]) -> Box:
    all_dims = []
    for f, e in zip(values, errs):
        for dim in get_default_limits(f, None, log_dims, e).vector.item_names:
            if dim not in all_dims and dim not in ignore_dims:
                all_dims.append(dim)
    if '_' in all_dims and len(all_dims) > 2:
        all_dims.remove('_')
    all_bounds = [embed(get_default_limits(f, all_dims, log_dims, e).without(ignore_dims.names).largest(shape), all_dims) for f, e in zip(values, errs)]
    bounds: Box = math.stack(all_bounds, batch('_fields'), simplify=True)
    lower = math.finite_min(bounds.lower, bounds.shape.without('vector'), default=-math.INF)
    upper = math.finite_max(bounds.upper, bounds.shape.without('vector'), default=math.INF)
    return Box(lower, upper)


def _insert_value_dim(space: Box, pos: Tuple[int, int], subplots: dict, min_val, max_val):
    row, col = pos
    axis = space.vector.item_names[0]
    new_axis = Box(_=(min_val, max_val))
    if space.vector.size <= 1:
        for (r, c), other_space in subplots.items():
            dims: tuple = other_space.vector.item_names
            if r == row and axis in dims and len(dims) == 2 and dims.index(axis) == 1:
                return concat([new_axis, space], 'vector')  # values along X
        return concat([space, new_axis], 'vector')  # values along Y (standard)
    elif space.vector.size > 2 and '_' in space.vector.item_names:
        others = [d for d in space.vector.item_names if d != '_']
        return space.vector[others]
    else:
        return space


def layout_color(content: Dict[Tuple[int, int], List[Field]], indices: Dict[Tuple[int, int], List[dict]], color: Tensor):
    with math.NUMPY:
        result = {}
        for pos, fields in content.items():
            result_pos = result[pos] = []
            counter = 0
            for i, f in enumerate(fields):
                idx = indices[pos][i]
                if (color[idx] != None).all:  # user-specified color
                    result_pos.append(color[idx])
                else:
                    cmap: bool = requires_color_map(f)
                    channels = channel(f).without('vector')
                    channel_colors = counter + math.range_tensor(channels)
                    result_pos.append(wrap('cmap') if cmap else channel_colors)
                    if not cmap:
                        counter += channels.volume
        return result


def overlay(*fields: Union[Field, Tensor, Geometry]) -> Tensor:
    """
    Specify that multiple fields should be drawn on top of one another in the same figure.
    The fields will be plotted in the order they are given, i.e. the last field on top.

    >>> plot(vis.overlay(heatmap, points, velocity))

    Args:
        *fields: `Field` or `Tensor` instances

    Returns:
        Plottable object
    """
    return layout(fields, math.channel('overlay'))


def write_image(path: str, figure=None, dpi=120., close=False, transparent=True):
    """
    Save a figure to an image file.

    Args:
        figure: Matplotlib or Plotly figure or text.
        path: File path.
        dpi: Pixels per inch.
        close: Whether to close the figure after saving it.
        transparent: Whether to save the figure with transparent background.
    """
    figure = figure or LAST_FIGURE[0]
    if figure is None:
        warnings.warn("No plot yet created with phi.vis; nothing to save.", RuntimeWarning)
        return
    assert figure is not None, "No figure to save."
    lib = get_plots_by_figure(figure)
    path = os.path.expanduser(path)
    directory = os.path.abspath(os.path.dirname(path))
    os.path.isdir(directory) or os.makedirs(directory)
    lib.save(figure, path, dpi, transparent)
    if close:
        close_(figure=figure)


def default_gui() -> Gui:
    if GUI_OVERRIDES:
        return GUI_OVERRIDES[-1]
    if is_jupyter():
        raise NotImplementedError("There is currently no GUI support for Python notebooks. Use `vis.plot()` to display plots or animations instead.")
    else:
        options = ['dash', 'console']
    for option in options:
        try:
            return get_gui(option)
        except ImportError as import_error:
            warnings.warn(f"{option} user interface is unavailable because of missing dependency: {import_error}.", ImportWarning)
    raise RuntimeError("No user interface available.")


def get_gui(gui: Union[str, Gui]) -> Gui:
    if GUI_OVERRIDES:
        return GUI_OVERRIDES[-1]
    if isinstance(gui, str):
        if gui == 'dash':
            from ._dash.dash_gui import DashGui
            return DashGui()
        elif gui == 'console':
            from ._console import ConsoleGui
            return ConsoleGui()
        else:
            raise NotImplementedError(f"No display available with name {gui}")
    elif isinstance(gui, Gui):
        return gui
    else:
        raise ValueError(gui)


GUI_OVERRIDES = []


@contextmanager
def force_use_gui(gui: Gui):
    GUI_OVERRIDES.append(gui)
    try:
        yield None
    finally:
        assert GUI_OVERRIDES.pop(-1) is gui


_LOADED_PLOTTING_LIBRARIES: List[PlottingLibrary] = []


def default_plots(content: Dict[Tuple[int, int], List[Field]]) -> PlottingLibrary:
    is_3d = False
    for fields in content.values():
        if any(f.spatial_rank == 3 for f in fields):
            is_3d = True
            break
    if is_jupyter():
        options = ['plotly', 'matplotlib'] if is_3d else ['matplotlib', 'plotly']
    else:
        options = ['plotly', 'matplotlib'] if is_3d else ['matplotlib', 'plotly', 'ascii']
    for option in options:
        try:
            return get_plots(option)
        except ImportError as import_error:
            warnings.warn(f"{option} user interface is unavailable because of missing dependency: {import_error}.", ImportWarning)
    raise RuntimeError("No user interface available.")


def get_plots(lib: Union[str, PlottingLibrary]) -> PlottingLibrary:
    if isinstance(lib, PlottingLibrary):
        return lib
    for loaded_lib in _LOADED_PLOTTING_LIBRARIES:
        if loaded_lib.name == lib:
            return loaded_lib
    if lib == 'matplotlib':
        from ._matplotlib._matplotlib_plots import MATPLOTLIB
        _LOADED_PLOTTING_LIBRARIES.append(MATPLOTLIB)
        return MATPLOTLIB
    elif lib == 'plotly':
        from ._dash._plotly_plots import PLOTLY
        _LOADED_PLOTTING_LIBRARIES.append(PLOTLY)
        return PLOTLY
    elif lib == 'ascii':
        from ._console._console_plot import CONSOLE
        _LOADED_PLOTTING_LIBRARIES.append(CONSOLE)
        return CONSOLE
    else:
        raise NotImplementedError(f"No plotting library available with name {lib}")


def get_plots_by_figure(figure: Union[Tensor, Any]):
    if isinstance(figure, Tensor):
        figure = next(iter(figure))
    for loaded_lib in _LOADED_PLOTTING_LIBRARIES:
        if loaded_lib.is_figure(figure):
            return loaded_lib
    else:
        raise ValueError(f"No library found matching figure {figure} from list {_LOADED_PLOTTING_LIBRARIES}")


def share_axes(*lims: Box, axes: Tuple[str]) -> Box or None:
    lower = {}
    upper = {}
    for axis in axes:
        if any(axis in box.vector.item_names for box in lims):
            lower[axis] = math.min([box.lower.vector[axis] for box in lims if axis in box.vector.item_names], shape)
            upper[axis] = math.max([box.upper.vector[axis] for box in lims if axis in box.vector.item_names], shape)
    return Box(vec(**lower), vec(**upper)) if lower else None


def replace_bounds(box: Box, replace: Box):
    if replace is None:
        return box
    lower = {axis: replace.lower.vector[axis] if axis in replace.vector.item_names else box.lower.vector[axis] for axis in box.vector.item_names}
    upper = {axis: replace.upper.vector[axis] if axis in replace.vector.item_names else box.upper.vector[axis] for axis in box.vector.item_names}
    return Box(vec(**lower), vec(**upper))


def _title(obj: Tensor, idx: dict):
    obj = obj[idx]
    while isinstance(obj, Layout) and not obj.shape and isinstance(obj.native(), Tensor):
        obj = obj.native()[idx]
    if not obj.shape:
        return obj.native()
    return ", ".join(obj)
