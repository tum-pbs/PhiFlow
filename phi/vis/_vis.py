import inspect
import os
import sys
import warnings
from contextlib import contextmanager
from threading import Thread
from typing import Tuple, List, Dict, Union

from ._user_namespace import get_user_namespace, UserNamespace, DictNamespace
from ._viewer import create_viewer, Viewer
from ._vis_base import Control, value_range, Action, VisModel, Gui, PlottingLibrary, common_index, to_field
from ._vis_base import title_label
from .. import math
from ..field import SampledField, Scene, Field
from ..field._scene import _slugify_filename
from ..geom import Geometry, Box, embed
from ..math import Tensor, layout, batch, Shape
from ..math import wrap
from ..math._shape import parse_dim_order, DimFilter, EMPTY_SHAPE, merge_shapes


def show(*model: Union[VisModel, SampledField, tuple, list, Tensor, Geometry],
         play=True,
         gui: Union[Gui, str] = None,
         lib: Union[Gui, str] = None,
         keep_alive=True,
         **config):
    """
    If `model` is a user interface model, launches the registered user interface.
    This will typically be the Dash web interface or the console interface if dash is not available.
    This method prepares the `model` before showing it. No more fields should be added to the vis after this method is invoked.

    See Also:
        `view()`.

    If `model` is plottable, e.g. a `SampledField` or `Tensor`, a figure is created and shown.
    If `model` is a figure, it is simply shown.

    See Also:
        `plot()`.

    This method may block until the GUI or plot window is closed.

    Also see the user interface documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html

    Args:
      model: (Optional) `VisModel`, the application or plottable object to display.
        If unspecified, shows the most recently plotted figure.
      play: If true, invokes `App.play()`. The default value is False unless "autorun" is passed as a command line argument.
      gui: Deprecated. Use `lib` instead. (optional) class of GUI to use
      lib: Gui class or plotting library as `str`, e.g. `'matplotlib'` or `'plotly'`
      keep_alive: Whether the GUI keeps the vis alive. If `False`, the program will exit when the main script is finished.
      **config: additional GUI configuration parameters.
        For a full list of parameters, see the respective GUI documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html
    """
    lib = lib if lib is not None else gui
    if len(model) == 1 and isinstance(model[0], VisModel):
        model[0].prepare()
        # --- Setup Gui ---
        gui = default_gui() if lib is None else get_gui(lib)
        gui.configure(config)
        gui.setup(model[0])
        if play:  # this needs to be done even if model cannot progress right now
            gui.auto_play()
        if gui.asynchronous:
            display_thread = Thread(target=lambda: gui.show(True), name="AsyncGui", daemon=not keep_alive)
            display_thread.start()
        else:
            gui.show(True)  # may be blocking call
    elif len(model) == 0:
        plots = default_plots() if lib is None else get_plots(lib)
        return plots.show(plots.current_figure)
    else:
        plots = default_plots() if lib is None else get_plots(lib)
        fig_tensor = plot(*model, lib=plots, **config)
        if isinstance(fig_tensor, Tensor):
            for fig in fig_tensor:
                plots.show(fig)
        else:
            return plots.show(fig_tensor)


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



RECORDINGS = {}


def record(*fields: Union[str, SampledField]) -> Viewer:
    user_namespace = get_user_namespace(1)
    variables = _default_field_variables(user_namespace, fields)
    viewer = create_viewer(user_namespace, variables, "record", "", scene=None, asynchronous=False, controls=(),
                           actions={}, log_performance=False)
    viewer.post_step.append(lambda viewer: print(viewer.steps, end=" "))
    viewer.progress_unavailable.append(lambda viewer: print())
    return viewer


def view(*fields: Union[str, SampledField],
         play: bool = True,
         gui=None,
         name: str = None,
         description: str = None,
         scene: Union[bool, Scene] = False,
         keep_alive=True,
         select: Union[str, tuple, list] = '',
         framerate=None,
         namespace=None,
         log_performance=True,
         **config) -> Viewer:
    """
    Show `fields` in a graphical user interface.

    `fields` may contain instances of `Field` or variable names of top-level variables (main module or Jupyter notebook).
    During loops, e.g. `view().range()`, the variable status is tracked and the GUI is updated.

    When called from a Python script, name and description may be specified in the module docstring (string before imports).
    The first line is interpreted as the name, the rest as the subtitle.
    If not specified, a generic name and description is chosen.

    Args:
        *fields: (Optional) Contents to be displayed. Either variable names or values.
            For field instances, all variables referencing the value will be shown.
            If not provided, the user namespace is searched for Field variables.
        play: Whether to immediately start executing loops.
        gui: (Optional) Name of GUI as `str` or GUI class.
            Built-in GUIs can be selected via `'dash'`, `'console'`.
            See https://tum-pbs.github.io/PhiFlow/Visualization.html
        name: (Optional) Name to display in GUI and use for the output directory if `scene=True`.
            Will be generated from the top-level script if not provided.
        description: (Optional) Description to be displayed in the GUI.
            Will be generated from the top-level script if not provided.
        scene: Existing `Scene` to write into or `bool`. If `True`, creates a new Scene in `~/phi/<name>`
        keep_alive: Whether the GUI should keep running even after the main thread finishes.
        framerate: Target frame rate in Hz. Play will not step faster than the framerate. `None` for unlimited frame rate.
        select: Dimension names along which one item to show is selected.
            Dimensions may be passed as `tuple` of `str` or as comma-separated names in a single `str`.
            For each `select` dimension, an associated selection slider will be created.
        log_performance: Whether to measure and log the time each step takes.
            If `True`, will be logged as `step_time` to `log_step_time.txt`.
        **config: Additional GUI configuration arguments.

    Returns:
        `Viewer`
    """
    default_namespace = get_user_namespace(1)
    user_namespace = default_namespace if namespace is None else DictNamespace(namespace,
                                                                               title=default_namespace.get_title(),
                                                                               description=default_namespace.get_description(),
                                                                               reference=default_namespace.get_reference())
    variables = _default_field_variables(user_namespace, fields)
    actions = dict(ACTIONS)
    ACTIONS.clear()
    if scene is False:
        scene = None
    elif scene is True:
        scene = Scene.create(os.path.join("~", "phi", _slugify_filename(name or user_namespace.get_reference())))
        print(f"Created scene at {scene}")
    else:
        assert isinstance(scene, Scene)
    name = name or user_namespace.get_title()
    description = description or user_namespace.get_description()
    gui = default_gui() if gui is None else get_gui(gui)
    controls = tuple(c for c in sorted(CONTROL_VARS.values(), key=lambda c: c.name) if
                     user_namespace.get_variable(c.name) is not None)
    CONTROL_VARS.clear()
    viewer = create_viewer(user_namespace, variables, name, description, scene, asynchronous=gui.asynchronous,
                           controls=controls, actions=actions, log_performance=log_performance)
    show(viewer, play=play, gui=gui, keep_alive=keep_alive, framerate=framerate, select=select, **config)
    return viewer


def _default_field_variables(user_namespace: UserNamespace, fields: tuple):
    names = []
    values = []
    if len(fields) == 0:  # view all Fields
        user_variables = user_namespace.list_variables(only_public=True, only_current_scope=True)
        for name, val in user_variables.items():
            if isinstance(val, SampledField):
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


def plot(*fields: Union[SampledField, Tensor, Geometry],
         lib: Union[str, PlottingLibrary] = None,
         row_dims: DimFilter = None,
         col_dims: DimFilter = batch,
         animate: DimFilter = None,
         overlay: DimFilter = 'overlay',
         title: Union[str, Tensor] = None,
         size=(12, 5),
         same_scale=True,
         log_dims: Union[str, tuple, list, Shape] = '',
         show_color_bar=True,
         color: Union[str, int, Tensor] = None,
         alpha: Union[float, Tensor] = 1.,
         err: Union[Tensor, tuple, list, float] = 0.,
         frame_time=100,
         repeat=True):
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
        title: String `Tensor` with dimensions `rows` and `cols`.
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
    positioning = {}
    indices: Dict[Tuple[int, int], List[dict]] = {}
    nrows, ncols, fig_shape, reduced_shape = layout_sub_figures(layout(fields, batch('args')), row_dims, col_dims, animate, overlay, 0, 0, positioning, indices, {})
    animate = fig_shape.only(animate)
    fig_shape = fig_shape.without(animate)
    plots = default_plots() if lib is None else get_plots(lib)
    # --- Process arguments ---
    if same_scale:
        if any([f.values.dtype.kind == complex for l in positioning.values() for f in l]):
            min_val = 0
            max_val = max([float(abs(f.values).finite_max) for l in positioning.values() for f in l])
        else:
            min_val = min([float(f.values.finite_min) for l in positioning.values() for f in l])
            max_val = max([float(f.values.finite_max) for l in positioning.values() for f in l])
    else:
        min_val = max_val = None
    subplots = {pos: _space(*fields, ignore_dims=animate) for pos, fields in positioning.items()}
    if isinstance(title, str):
        title = {pos: title for pos in positioning}
    elif isinstance(title, Tensor):
        title = {(row, col): title.rows[row].cols[col].native() for (row, col) in positioning}
    else:
        assert title is None, f"title must be a str or Tensor but got {title}"
        title = {pos: title_label(common_index(*i, exclude=reduced_shape.singleton)) for pos, i in indices.items()}
    log_dims = parse_dim_order(log_dims) or ()
    color = layout_pytree_node(color, wrap_leaf=True)
    alpha = layout_pytree_node(alpha, wrap_leaf=True)
    err = layout_pytree_node(err, wrap_leaf=True)
    # --- animate or plot ---
    if fig_shape.volume == 1:
        figure, axes = plots.create_figure(size, nrows, ncols, subplots, title, log_dims)
        if animate:
            def plot_frame(frame: int):
                for pos, fields in positioning.items():
                    for i, f in enumerate(fields):
                        idx = indices[pos][i]
                        f = f[{animate.name: frame}]
                        plots.plot(f, figure, axes[pos], subplots[pos], min_val, max_val, show_color_bar, color[idx], alpha[idx], err[idx])
                plots.finalize(figure)
            anim = plots.animate(figure, animate.size, plot_frame, frame_time, repeat)
            LAST_FIGURE[0] = anim
            plots.close(figure)
            return anim
        else:
            for pos, fields in positioning.items():
                for i, f in enumerate(fields):
                    idx = indices[pos][i]
                    plots.plot(f, figure, axes[pos], subplots[pos], min_val, max_val, show_color_bar, color[idx], alpha[idx], err[idx])
            plots.finalize(figure)
            LAST_FIGURE[0] = figure
            return layout(figure)
    else:
        raise NotImplementedError(f"Figure batches not yet supported. Use rows and cols to reduce all batch dimensions. Not reduced. {fig_shape}")


def layout_pytree_node(data, wrap_leaf=False):
    if isinstance(data, tuple):
        return layout(data, batch('tuple'))
    elif isinstance(data, list):
        return layout(data, batch('list'))
    elif isinstance(data, dict):
        return layout(data, batch('dict'))
    return wrap(data) if wrap_leaf else data


def layout_sub_figures(data: Union[Tensor, SampledField],
                       row_dims: DimFilter,
                       col_dims: DimFilter,
                       animate: DimFilter,  # do not reduce these dims, has priority
                       overlay: DimFilter,
                       offset_row: int,
                       offset_col: int,
                       positioning: Dict[Tuple[int, int], List],
                       indices: Dict[Tuple[int, int], List[dict]],
                       base_index: Dict[str, Union[int, str]]) -> Tuple[int, int, Shape, Shape]:  # rows, cols
    if data is None:
        raise ValueError(f"Cannot layout figure for '{data}'")
    data = layout_pytree_node(data, wrap_leaf=False)
    if isinstance(data, Tensor) and data.dtype.kind == object:  # layout
        rows, cols = 0, 0
        non_reduced = math.EMPTY_SHAPE
        dim0 = reduced = data.shape[0]
        if dim0.only(overlay):
            for overlay_index in dim0.only(overlay).meshgrid(names=True):  # overlay these fields
                e_rows, e_cols, d_non_reduced, d_reduced = layout_sub_figures(data[overlay_index].native(), row_dims, col_dims, animate, overlay, offset_row, offset_col, positioning, indices, {**base_index, **overlay_index})
                rows = max(rows, e_rows)
                cols = max(cols, e_cols)
                non_reduced &= d_non_reduced
                reduced = merge_shapes(reduced, d_reduced, allow_varying_sizes=True)
        elif dim0.only(animate):
            data = math.stack(data.native(), dim0)
            return layout_sub_figures(data, row_dims, col_dims, animate, overlay, offset_row, offset_col, positioning, indices, base_index)
        else:
            elements = data.unstack(dim0.name)
            for item_name, e in zip(dim0.get_item_names(dim0.name) or range(dim0.size), elements):
                index = dict(base_index, **{dim0.name: item_name})
                if dim0.only(row_dims):
                    e_rows, e_cols, e_non_reduced, e_reduced = layout_sub_figures(e.native(), row_dims, col_dims, animate, overlay, offset_row + rows, offset_col, positioning, indices, index)
                    rows += e_rows
                    cols = max(cols, e_cols)
                elif dim0.only(col_dims):
                    e_rows, e_cols, e_non_reduced, e_reduced = layout_sub_figures(e.native(), row_dims, col_dims, animate, overlay, offset_row, offset_col + cols, positioning, indices, index)
                    cols += e_cols
                    rows = max(rows, e_rows)
                else:
                    e_rows, e_cols, e_non_reduced, e_reduced = layout_sub_figures(e.native(), row_dims, col_dims, animate, overlay, offset_row, offset_col, positioning, indices, index)
                    cols = max(cols, e_cols)
                    rows = max(rows, e_rows)
                non_reduced &= e_non_reduced
                reduced = merge_shapes(reduced, e_reduced, allow_varying_sizes=True)
        return rows, cols, non_reduced, reduced
    else:
        data = to_field(data)
        overlay = data.shape.only(overlay)
        animate = data.shape.only(animate).without(overlay)
        row_shape = data.shape.only(row_dims).without(animate).without(overlay)
        col_shape = data.shape.only(col_dims).without(row_dims).without(animate).without(overlay)
        non_reduced: Shape = batch(data).without(row_dims).without(col_dims) & animate
        for ri, r in enumerate(row_shape.meshgrid(names=True)):
            for ci, c in enumerate(col_shape.meshgrid(names=True)):
                for o in overlay.meshgrid(names=True):
                    sub_data = data[r][c][o]
                    positioning.setdefault((offset_row + ri, offset_col + ci), []).append(sub_data)
                    indices.setdefault((offset_row + ri, offset_col + ci), []).append(dict(base_index, **r, **c, **o))
        return row_shape.volume, col_shape.volume, non_reduced, EMPTY_SHAPE


def _space(*values: Field or Tensor, ignore_dims: Shape) -> Box:
    all_dims = []
    for f in values:
        for dim in f.bounds.vector.item_names:
            if dim not in all_dims and dim not in ignore_dims:
                all_dims.append(dim)
    all_bounds = [embed(f.bounds.without(ignore_dims.names), all_dims) for f in values]
    if len(all_bounds) == 1:
        return all_bounds[0]
    bounds: Box = math.stack(all_bounds, batch('_fields'))
    lower = math.finite_min(bounds.lower, '_fields', default=-math.INF)
    upper = math.finite_max(bounds.upper, '_fields', default=math.INF)
    return Box(lower, upper)


def overlay(*fields: Union[SampledField, Tensor]) -> Tensor:
    """
    Specify that multiple fields should be drawn on top of one another in the same figure.
    The fields will be plotted in the order they are given, i.e. the last field on top.

    >>> plot(vis.overlay(heatmap, points, velocity))

    Args:
        *fields: `SampledField` or `Tensor` instances

    Returns:
        Plottable object
    """
    return layout(fields, math.channel('overlay'))


def write_image(path: str, figure=None, dpi=120.):
    """
    Save a figure to an image file.

    Args:
        figure: Matplotlib or Plotly figure or text.
        path: File path.
        dpi: Pixels per inch.
    """
    figure = figure or LAST_FIGURE[0]
    if figure is None:
        figure = default_plots().current_figure
    assert figure is not None, "No figure to save."
    lib = get_plots_by_figure(figure)
    lib.save(figure, path, dpi)


def default_gui() -> Gui:
    if GUI_OVERRIDES:
        return GUI_OVERRIDES[-1]
    if 'google.colab' in sys.modules or 'ipykernel' in sys.modules:
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


def default_plots() -> PlottingLibrary:
    if 'google.colab' in sys.modules or 'ipykernel' in sys.modules:
        options = ['matplotlib']
    else:
        options = ['matplotlib', 'plotly', 'ascii']
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


def get_plots_by_figure(figure):
    for loaded_lib in _LOADED_PLOTTING_LIBRARIES:
        if loaded_lib.is_figure(figure):
            return loaded_lib
    else:
        raise ValueError(f"No library found matching figure {figure} from list {_LOADED_PLOTTING_LIBRARIES}")

