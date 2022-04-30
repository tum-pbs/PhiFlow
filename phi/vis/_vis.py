import inspect
import os
import sys
import warnings
from contextlib import contextmanager
from threading import Thread
from typing import Tuple, List, Dict, Callable

from ._user_namespace import get_user_namespace, UserNamespace, DictNamespace
from ._viewer import create_viewer, Viewer
from ._vis_base import Control, value_range, Action, VisModel, Gui, \
    PlottingLibrary
from .. import math, field
from ..field import SampledField, Scene, Field, PointCloud
from ..field._scene import _slugify_filename
from ..geom import Geometry
from ..math import Tensor, layout, batch, Shape
from ..math._tensors import Layout


def show(*model: VisModel or SampledField or tuple or list or Tensor or Geometry,
         play=True,
         gui: Gui or str = None,
         keep_alive=True,
         **config):
    """
    If `model` is a user interface model, launches the registered user interface.
    This will typically be the widgets interface for Jupyter or Colab notebooks, the Dash web interface for scripts or the console interface if the above are not available.
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
      gui: (optional) class of GUI to use
      keep_alive: Whether the GUI keeps the vis alive. If `False`, the program will exit when the main script is finished.
      **config: additional GUI configuration parameters.
        For a full list of parameters, see the respective GUI documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html
    """
    if len(model) == 1 and isinstance(model[0], VisModel):
        model[0].prepare()
        # --- Setup Gui ---
        gui = default_gui() if gui is None else get_gui(gui)
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
        plots = default_plots() if gui is None else get_plots(gui)
        return plots.show(plots.current_figure)
    else:
        plots = default_plots() if gui is None else get_plots(gui)
        fig = plot(*model, lib=plots, **config)
        return plots.show(fig)


RECORDINGS = {}


def record(*fields: str or SampledField) -> Viewer:
    user_namespace = get_user_namespace(1)
    variables = _default_field_variables(user_namespace, fields)
    viewer = create_viewer(user_namespace, variables, "record", "", scene=None, asynchronous=False, controls=(),
                           actions={}, log_performance=False)
    viewer.post_step.append(lambda viewer: print(viewer.steps, end=" "))
    viewer.progress_unavailable.append(lambda viewer: print())
    return viewer


def view(*fields: str or SampledField,
         play: bool = True,
         gui=None,
         name: str = None,
         description: str = None,
         scene: bool or Scene = False,
         keep_alive=True,
         select: str or tuple or list = '',
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
            Built-in GUIs can be selected via `'dash'`, `'console'` and `'widgets'`.
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
    ```python
    dt = control(1.0, (0.1, 10), name="Time increment (dt)")
    ```

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


LAST_FIGURE = [None]


def plot(*fields: SampledField or Tensor or Layout,
         lib: str or PlottingLibrary = None,
         row_dims: str or Shape or tuple or list or Callable = None,
         col_dims: str or Shape or tuple or list or Callable = batch,
         animate: str or Shape or tuple or list or Callable = None,
         title: Tensor = None,
         size=(12, 5),
         same_scale=True,
         show_color_bar=True,
         frame_time=200,
         repeat=True,
         **plt_args):
    """
    Creates one or multiple figures and sub-figures and plots the given fields.

    To show the figures, use `show()`.

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
        show_color_bar: Whether to display color bars for heat maps.
        animate: Time dimension to animate.
            If not present in the data, will produce a regular plot instead.
        frame_time: Interval between frames in the animation.
        repeat: Whether the animation should loop.

    Returns:
        If all batch dimensions are reduced by `rows` and `cols`, returns a single figure created for `data`.
        Otherwise returns a `Tensor` of figures (not yet supported).
    """
    positioning = {}
    nrows, ncols, fig_shape = layout_sub_figures(math.layout(fields, batch('args')), row_dims, col_dims, animate, 0, 0, positioning)
    animate = fig_shape.only(animate)
    fig_shape = fig_shape.without(animate)
    plots = default_plots() if lib is None else get_plots(lib)
    if same_scale:
        if any([f.values.dtype.kind == complex for l in positioning.values() for f in l]):
            min_val = 0
            max_val = float(max([abs(f.values).max for l in positioning.values() for f in l]))
        else:
            min_val = float(min([f.values.min for l in positioning.values() for f in l]))
            max_val = float(max([f.values.max for l in positioning.values() for f in l]))
    else:
        min_val = max_val = None
    subplots = {pos: max([f.spatial_rank for f in fields]) for pos, fields in positioning.items()}
    if title is None or isinstance(title, str):
        title = layout(title)
    assert isinstance(title, Tensor), "title must be a Tensor or str"
    if fig_shape.volume == 1:
        figure, axes = plots.create_figure(size, nrows, ncols, subplots, title)
        plots.plotting_done(figure, axes)
        if animate:
            def plot_frame(frame: int):
                for pos, fields in positioning.items():
                    for f in fields:
                        f = f[{animate.name: frame}]
                        plots.plot(f, figure, axes[pos], min_val=min_val, max_val=max_val, show_color_bar=show_color_bar, **plt_args)
            anim = plots.animate(figure, animate.size, plot_frame, frame_time, repeat)
            LAST_FIGURE[0] = anim
            plots.plotting_done(figure, axes)
            return anim
        else:
            for pos, fields in positioning.items():
                for f in fields:
                    plots.plot(f, figure, axes[pos], min_val=min_val, max_val=max_val, show_color_bar=show_color_bar, **plt_args)
            LAST_FIGURE[0] = figure
            plots.plotting_done(figure, axes)
            return figure
    else:
        raise NotImplementedError(f"Figure batches not yet supported. Use rows and cols to reduce all batch dimensions. Not reduced. {fig_shape}")


def layout_sub_figures(data: Tensor or Layout or SampledField,
                       row_dims: str or Shape or tuple or list or Callable,
                       col_dims: str or Shape or tuple or list or Callable,
                       animate: str or Shape or tuple or list or Callable,  # do not reduce these dims, has priority
                       offset_row: int,
                       offset_col: int,
                       positioning: Dict[Tuple[int, int], List]) -> Tuple[int, int, Shape]:  # rows, cols
    if data is None:
        raise ValueError(f"Cannot layout figure for '{data}'")
    if isinstance(data, list):
        data = math.layout(data, batch('list'))
    elif isinstance(data, tuple):
        data = math.layout(data, batch('tuple'))
    if isinstance(data, Layout):
        rows, cols = 0, 0
        non_reduced = math.EMPTY_SHAPE
        if not batch(data):  # overlay
            for d in data:  # overlay these fields
                e_rows, e_cols, d_non_reduced = layout_sub_figures(d, row_dims, col_dims, animate, offset_row, offset_col, positioning)
                rows = max(rows, e_rows)
                cols = max(cols, e_cols)
                non_reduced &= d_non_reduced
        else:
            dim0 = data.shape[0]
            elements = data.unstack(dim0.name)
            for e in elements:
                if dim0.only(animate):
                    # assert data.shape.rank == 1,
                    raise NotImplementedError("To animate plots, stack data along time axis first.")
                elif dim0.only(row_dims):
                    e_rows, e_cols, e_non_reduced = layout_sub_figures(e.native(), row_dims, col_dims, animate, offset_row + rows, offset_col, positioning)
                    rows += e_rows
                    cols = max(cols, e_cols)
                elif dim0.only(col_dims):
                    e_rows, e_cols, e_non_reduced = layout_sub_figures(e.native(), row_dims, col_dims, animate, offset_row, offset_col + cols, positioning)
                    cols += e_cols
                    rows = max(rows, e_rows)
                else:
                    e_rows, e_cols, e_non_reduced = layout_sub_figures(e.native(), row_dims, col_dims, animate, offset_row, offset_col, positioning)
                    cols = max(cols, e_cols)
                    rows = max(rows, e_rows)
                non_reduced &= e_non_reduced
        return rows, cols, non_reduced
    else:
        if isinstance(data, Tensor):
            data = field.tensor_as_field(data)
        elif isinstance(data, Geometry):
            data = PointCloud(data)
        assert isinstance(data, Field), data
        animate = data.shape.only(animate)
        row_shape = batch(data).only(row_dims).without(animate)
        col_shape = batch(data).only(col_dims).without(row_dims).without(animate)
        non_reduced: Shape = batch(data).without(row_dims).without(col_dims) & animate
        for ri, r in enumerate(row_shape.meshgrid()):
            for ci, c in enumerate(col_shape.meshgrid()):
                sub_data = data[r][c]
                positioning.setdefault((offset_row + ri, offset_col + ci), []).append(sub_data)
        return row_shape.volume, col_shape.volume, non_reduced


    # if len(plottable) > 1:
    #     plottable
    #
    # return math.layout([], math.channel('cols, rows, overlay'))
    # else:  # x, y, z
    #     if channel in value.shape.spatial and 'vector' in value.shape:
    #         return value.vector[channel]
    #     elif 'vector' in value.shape:
    #         raise ValueError(
    #             f"No {channel} component present. Available dimensions: {', '.join(value.shape.spatial.names)}")
    #     else:
    #         return value


def overlay(*fields: SampledField or Tensor) -> Tensor:
    """
    Specify that multiple fields should be drawn on top of one another in the same figure.
    The fields will be plotted in the order they are given, i.e. the last field on top.

    ```python
    vis.plot(vis.overlay(heatmap, points, velocity))
    ```

    Args:
        *fields: `SampledField` or `Tensor` instances

    Returns:
        Plottable object
    """
    return math.layout(fields, math.channel('overlay'))


def write_image(path: str, figure=None, dpi=120.):
    """
    Save a figure to an image file.

    Args:
        figure: Matplotlib or Plotly figure or text.
        path: File path.
        dpi: Pixels per inch.
    """
    figure = figure or LAST_FIGURE[0]
    lib = get_plots_by_figure(figure)
    lib.save(figure, path, dpi)


def default_gui() -> Gui:
    if GUI_OVERRIDES:
        return GUI_OVERRIDES[-1]
    if 'google.colab' in sys.modules or 'ipykernel' in sys.modules:
        options = ['widgets']
    else:
        options = ['dash', 'console']
    for option in options:
        try:
            return get_gui(option)
        except ImportError as import_error:
            warnings.warn(f"{option} user interface is unavailable because of missing dependency: {import_error}.", ImportWarning)
    raise RuntimeError("No user interface available.")


def get_gui(gui: str or Gui) -> Gui:
    if GUI_OVERRIDES:
        return GUI_OVERRIDES[-1]
    if isinstance(gui, str):
        if gui == 'dash':
            from ._dash.dash_gui import DashGui
            return DashGui()
        elif gui == 'console':
            from ._console import ConsoleGui
            return ConsoleGui()
        # elif gui == 'matplotlib':
        #     from ._matplotlib.matplotlib_gui import MatplotlibGui
        #     return MatplotlibGui()
        elif gui == 'widgets':
            from ._widgets import WidgetsGui
            return WidgetsGui()
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


def get_plots(lib: str or PlottingLibrary) -> PlottingLibrary:
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

