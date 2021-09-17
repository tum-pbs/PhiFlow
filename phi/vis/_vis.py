import inspect
import os
from threading import Thread

from ._user_namespace import get_user_namespace, UserNamespace, DictNamespace
from ._viewer import create_viewer, Viewer
from ._vis_base import get_gui, default_gui, Control, display_name, value_range, Action, VisModel, Gui
from ..field import SampledField, Scene
from ..field._scene import _slugify_filename


def show(model: VisModel or None = None, play=True, gui: Gui or str = None, keep_alive=True, **config):
    """
    Launch the registered user interface (web interface by default).

    This method may block until the GUI is closed.

    This method prepares the vis before showing it. No more fields should be added to the vis after this method is invoked.

    Also see the user interface documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html

    Args:
      model: (Optional) `VisModel`, the application to display. If unspecified, searches the calling script for a subclass of App and instantiates it.
      play: If true, invokes `App.play()`. The default value is False unless "autorun" is passed as a command line argument.
      gui: (optional) class of GUI to use
      keep_alive: Whether the GUI keeps the vis alive. If `False`, the program will exit when the main script is finished.
      **config: additional GUI configuration parameters.
        For a full list of parameters, see the respective GUI documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html
    """
    if model is None:
        import pylab
        pylab.show()
        return
    assert isinstance(model, VisModel), f"show() first argument must be an App instance but got {model}"
    model.prepare()
    # --- Setup Gui ---
    gui = default_gui() if gui is None else get_gui(gui)
    gui.configure(config)
    gui.setup(model)
    if play:  # this needs to be done even if model cannot progress right now
        gui.auto_play()
    if gui.asynchronous:
        display_thread = Thread(target=lambda: gui.show(True), name="AsyncGui", daemon=not keep_alive)
        display_thread.start()
    else:
        gui.show(True)  # may be blocking call


RECORDINGS = {}


def record(*fields: str or SampledField) -> Viewer:
    user_namespace = get_user_namespace(1)
    variables = _default_field_variables(user_namespace, fields)
    viewer = create_viewer(user_namespace, variables, "record", "", scene=None, asynchronous=False, controls=(), actions={}, log_performance=False)
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
        **config: Additional GUI configuration arguments.

    Returns:
        `Viewer`
    """
    default_namespace = get_user_namespace(1)
    user_namespace = default_namespace if namespace is None else DictNamespace(namespace, title=default_namespace.get_title(), description=default_namespace.get_description(), reference=default_namespace.get_reference())
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
    controls = tuple(c for c in sorted(CONTROL_VARS.values(), key=lambda c: c.name) if user_namespace.get_variable(c.name) is not None)
    CONTROL_VARS.clear()
    viewer = create_viewer(user_namespace, variables, name, description, scene, asynchronous=gui.asynchronous, controls=controls, actions=actions, log_performance=True)
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


def control(value, range: tuple = None, description="", **kwargs):
    """
    Mark a variable as controllable by any GUI created via `view()`.

    Example:
    ```python
    dt = control(1.0, (0.1, 10), name="Time increment")
    ```

    The value o

    Args:
        value: Initial value. Must be either `int`, `float´, `bool` or `str`.
        range: (Optional) Specify range of possible values as `(min, max)`. Only for `int` and `float` values.
        description: Description of what the control does.
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


CONTROL_VARS = {}


def action(fun):
    doc = inspect.getdoc(fun)
    ACTIONS[Action(fun.__name__, doc)] = fun


ACTIONS = {}
