import threading
import time
from collections import namedtuple
from math import log10
from threading import Lock
from typing import Tuple, Any, Optional, Dict, Callable, Union

from phi import field, math
from phi.field import SampledField, Scene, PointCloud, CenteredGrid
from phi.field._field_math import data_bounds
from phi.geom import Box, Cuboid, Geometry, Point
from phiml.math import Shape, EMPTY_SHAPE, Tensor, spatial, instance, wrap, channel, expand, non_batch

Control = namedtuple('Control', [
    'name',
    'control_type',  # type (float, int, str, bool)
    'initial',
    'value_range',  # (lo, hi) or ("
    'description',  # str
    'kwargs'  # dict
])

Action = namedtuple('Action', ['name', 'description'])


def value_range(control: Control) -> tuple:
    if control.control_type == float:
        if isinstance(control.value_range, tuple):
            assert len(control.value_range) == 2, f"Tuple must be (min, max) but got length {len(control.value_range)}"
            return control.value_range
        log_scale = is_log_control(control)
        if log_scale:
            magn = log10(control.initial)
            val_range = (10.0 ** (magn - 3.2), 10.0 ** (magn + 2.2))
        else:
            if control.initial == 0.0:
                val_range = (-10.0, 10.0)
            elif control.initial > 0:
                val_range = (0., 4. * control.initial)
            else:
                val_range = (2. * control.initial, -2. * control.initial)
    elif control.control_type == int:
        if isinstance(control.value_range, tuple):
            assert len(control.value_range) == 2, f"Tuple must be (min, max) but got length {len(control.value_range)}"
            return control.value_range
        if control.initial == 0:
            val_range = (-10, 10)
        elif control.initial > 0:
            val_range = (0, 4 * control.initial)
        else:
            val_range = (2 * control.initial, -2 * control.initial)
    elif control.control_type == bool:
        assert control.value_range is None, "Specifying range for bool controls is not allowed."
        return False, True
    elif control.control_type == str:
        if isinstance(control.value_range, tuple):
            return "", control.value_range
        return "", ""
    else:
        raise AssertionError(f"Not a numeric control: {control}")
    return val_range


def is_log_control(control: Control):
    if control.control_type != float:
        return False
    log_scale = control.kwargs.get('log')
    if log_scale is not None:
        return log_scale
    else:
        if control.value_range is None:
            return True
        else:
            if 0 in control.value_range:
                return False
            return control.value_range[1] / float(control.value_range[0]) > 10


class VisModel:

    def __init__(self, name: str = None, description: str = "", scene: Scene = None):
        self.start_time = time.time()
        """ Time of creation (`App` constructor invocation) """
        self.name = name if name is not None else self.__class__.__name__
        """ Human-readable name. """
        self.description = description
        """ Description to be displayed. """
        self.scene = scene
        """ Directory to which data and logging information should be written as `Scene` instance. """
        self.uses_existing_scene = scene.exist_properties() if scene is not None else False
        self.steps = 0
        """ Counts the number of times `step()` has been called. May be set by the user. """
        self.progress_lock = Lock()
        self.pre_step = []  # callback(vis)
        self.post_step = []  # callback(vis)
        self.progress_available = []  # callback(vis)
        self.progress_unavailable = []  # callback(vis)
        self.growing_dims = ()  # tuple or list, used by GUI to determine whether to scroll to last element
        self.message = None
        self.log_file = None

    def progress(self):
        pass

    @property
    def is_progressing(self) -> bool:
        return self.progress_lock.locked()

    @property
    def can_progress(self) -> bool:
        raise NotImplementedError(self)

    def prepare(self):
        pass

    @property
    def field_names(self) -> tuple:
        raise NotImplementedError(self)

    def get_field(self, name: str, dim_selection: dict) -> SampledField:
        """
        Returns the current value of a field.
        The name must be part of `VisModel.field_names`.

        Raises:
            `KeyError` if `field_name` is not a valid field.

        Args:
            name: Registered name of the field.
            dim_selection: Slices the field according to `selection`. `dict` mapping dimension names to `int` or `slice`.

        Returns:
            `SampledField`
        """
        raise NotImplementedError(self)

    def get_field_shape(self, name: str) -> Shape:
        value = self.get_field(name, {})
        if isinstance(value, (Tensor, SampledField)):
            return value.shape
        else:
            return EMPTY_SHAPE

    @property
    def curve_names(self) -> tuple:
        raise NotImplementedError(self)

    def get_curve(self, name: str) -> tuple:
        raise NotImplementedError(self)

    @property
    def controls(self) -> Tuple[Control]:
        raise NotImplementedError(self)

    def get_control_value(self, name):
        raise NotImplementedError(self)

    def set_control_value(self, name, value):
        raise NotImplementedError(self)

    @property
    def actions(self) -> Tuple[Action]:
        raise NotImplementedError(self)

    def run_action(self, name):
        raise NotImplementedError(self)

    # Implemented methods

    def _call(self, observers):
        for obs in observers:
            obs(self)


def get_control_by_name(model: VisModel, control_name: str):
    assert isinstance(control_name, str)
    for control in model.controls:
        if control.name == control_name:
            return control
    raise KeyError(f"No control with name '{control_name}'. Available controls: {model.controls}")


def _step_and_wait(model: VisModel, framerate=None):
    t = time.time()
    model.progress()
    if framerate is not None:
        remaining_time = 1.0 / framerate - (time.time() - t)
        if remaining_time > 0:
            time.sleep(remaining_time)


class AsyncPlay:

    def __init__(self, model: VisModel, max_steps, framerate):
        self.model = model
        self.max_steps = max_steps
        self.framerate = framerate
        self.paused = False
        self._finished = False

    def start(self):
        thread = threading.Thread(target=self, name='AsyncPlay')
        thread.start()

    def __call__(self):
        step_count = 0
        while not self.paused:
            _step_and_wait(self.model, framerate=self.framerate)
            step_count += 1
            if self.max_steps and step_count >= self.max_steps:
                break
        self._finished = True

    def pause(self):
        self.paused = True

    def __bool__(self):
        return not self._finished

    def __repr__(self):
        return status_message(self.model, self)


def status_message(model: VisModel, play_status: Union[AsyncPlay, None]):
    pausing = "/Pausing" if (play_status and play_status.paused) else ""
    current_action = "Running" if model.is_progressing else "Waiting"
    action = current_action if play_status else "Idle"
    message = f" - {model.message}" if model.message else ""
    return f"{action}{pausing} ({model.steps} steps){message}"


def play_async(model: VisModel, max_steps=None, framerate=None):
    """
    Run a number of steps.

    Args:
        model: Model to progress
        max_steps: (optional) stop when this many steps have been completed (independent of the `steps` variable) or `pause()` is called.
        framerate: Target frame rate in Hz.
    """
    assert model.can_progress
    play = AsyncPlay(model, max_steps, framerate)
    play.start()
    return play


def benchmark(model: VisModel, sequence_count):
    # self._pause = False  # TODO allow premature stop
    step_count = 0
    t = time.time()
    for i in range(sequence_count):
        model.progress()
        step_count += 1
        # if self._pause:
        #     break
    time_elapsed = time.time() - t
    return step_count, time_elapsed


class Gui:

    def __init__(self, asynchronous=False):
        """
        Creates a display for the given vis and initializes the configuration.
        This method does not set up the display. It only sets up the Gui object and returns as quickly as possible.
        """
        self.app: Optional[VisModel] = None
        self.asynchronous = asynchronous
        self.config = {}

    def configure(self, config: dict):
        """
        Updates the GUI configuration.
        This method may only be called while the GUI is not yet visible, i.e. before show() is called.

        Args:
            config: Complete or partial GUI-specific configuration. dictionary mapping from strings to JSON serializable values
        """
        self.config.update(config)

    def get_configuration(self) -> dict:
        """
        Returns the current configuration of the GUI.
        The returned dictionary may only contain serializable values and all keys must be strings.
        The configuration can be passed to another instance of this class using set_configuration().
        """
        return self.config

    def setup(self, app: VisModel):
        """
        Sets up all necessary GUI components.
        
        The GUI can register callbacks with the vis to be informed about vis-state changes induced externally.
        The vis can be assumed to be prepared when this method is called.
        
        This method is called after set_configuration() but before show()

        Args:
          app: vis to be displayed, may not be prepared or be otherwise invalid at this point.
        """
        self.app = app

    def show(self, caller_is_main: bool):
        """
        Displays the previously setup GUI.
        This method is blocking and returns only when the GUI is hidden.

        This method will always be called after setup().

        Args:
            caller_is_main: True if the calling script is the __main__ module.
        """
        pass

    def auto_play(self):
        """
        Called if `autorun=True`.
        If no Gui is specified, `App.run()` is called instead.
        """
        raise NotImplementedError(self)


class PlottingLibrary:

    def __init__(self, name: str, figure_classes: Union[tuple, list]):
        self.name = name
        self.figure_classes = tuple(figure_classes)
        self.current_figure = None
        self.recipes = []

    def __repr__(self):
        return self.name

    def is_figure(self, obj):
        if isinstance(obj, (tuple, list)):
            return isinstance(obj[0], self.figure_classes)
        return isinstance(obj, self.figure_classes)

    def create_figure(self,
                      size: tuple,
                      rows: int,
                      cols: int,
                      spaces: Dict[Tuple[int, int], Box],
                      titles: Dict[Tuple[int, int], str],
                      log_dims: Tuple[str, ...]) -> Tuple[Any, Dict[Tuple[int, int], Any]]:
        """
        Args:
            size: Figure size in inches.
            rows: Number of sub-figures laid out vertically.
            cols: Number of sub-figures laid out horizontally.
            spaces: Axes and range per sub-plot: `(x,y) -> Box`. Only subplot locations contained as keys should be plotted.
                To indicate automatic limit, the box will have a lower or upper limit of -inf or inf, respectively.
            titles: Subplot titles.
            log_dims: Dimensions along which axes should be log-scaled

        Returns:
            figure: Native figure object
            subfigures: Native sub-figures by subplot location.
        """
        raise NotImplementedError

    def animate(self, fig, frames: int, plot_frame_function: Callable, interval: float, repeat: bool):
        raise NotImplementedError

    def finalize(self, figure):
        raise NotImplementedError

    def close(self, figure):
        raise NotImplementedError

    def show(self, figure):
        raise NotImplementedError

    def save(self, figure, path: str, dpi: float):
        raise NotImplementedError

    def plot(self, data, figure, subplot, space, *args, **kwargs):
        for recipe in self.recipes:
            if recipe.can_plot(data, space):
                recipe.plot(data, figure, subplot, space, *args, **kwargs)
                return
        raise NotImplementedError(f"No {self.name} recipe found for {data}. Recipes: {self.recipes}")


class Recipe:

    def can_plot(self, data: SampledField, space: Box) -> bool:
        raise NotImplementedError

    def plot(self,
             data: SampledField,
             figure,
             subplot,
             space: Box,
             min_val: float,
             max_val: float,
             show_color_bar: bool,
             color: Tensor,
             alpha: Tensor,
             err: Tensor):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class GuiInterrupt(KeyboardInterrupt):
    pass


def gui_interrupt(*args, **kwargs):
    raise GuiInterrupt()


def display_name(python_name: Any):
    if isinstance(python_name, (int, bool)):
        return str(python_name)
    assert isinstance(python_name, str), f"name must be a str, int or bool but got {type(python_name)}"
    if python_name == '_':
        return ""
    n = list(python_name)
    n[0] = n[0].upper()
    for i in range(1, len(n)):
        if n[i] == "_":
            n[i] = " "
            if len(n) > i + 1:
                n[i + 1] = n[i + 1].upper()
    text = "".join(n)
    if "Reset" in text:
        return f"⏮ {text}"
    else:
        return text


def index_label(idx: dict, always_include_names: bool = False) -> Union[str, None]:
    if len(idx) == 0:
        return None
    if len(idx) == 1:
        if always_include_names:
            for name, value in idx.items():
                return f"{display_name(name)} {display_name(value)}"
        else:
            return display_name(next(iter(idx.values())))
    else:
        number_unlabelled_dims = len([1 for k, v in idx.items() if isinstance(v, int)])
        if number_unlabelled_dims <= 1:
            return " ".join([display_name(n) for n in idx.values()])
        else:
            return ", ".join(f'{k}={display_name(v)}' for k, v in idx.items())


def title_label(idx: dict):
    idx = {k: v for k, v in idx.items() if k not in ['tuple', 'list', 'dict', 'args'] or isinstance(v, str)}
    if len(idx) == 0:
        return None
    elif len(idx) == 1:
        for name, value in idx.items():
            if isinstance(value, int):
                return f"{display_name(name)} {display_name(value)}"
            else:
                return display_name(value)
    else:
        return index_label(idx)



def common_index(*indices: dict, exclude=()):
    return {k: v for k, v in indices[0].items() if k not in exclude and all([i[k] == v for i in indices])}


def select_channel(value: Union[SampledField, Tensor, tuple, list], channel: Union[str, None]):
    if isinstance(value, (tuple, list)):
        return [select_channel(v, channel) for v in value]
    if channel is None:
        return value
    elif channel == 'abs':
        if value.vector.exists:
            return field.vec_abs(value) if isinstance(value, SampledField) else math.vec_length(value)
        else:
            return value
    else:  # x, y, z
        if channel in value.shape.spatial and 'vector' in value.shape:
            return value.vector[channel]
        elif 'vector' in value.shape:
            raise ValueError(
                f"No {channel} component present. Available dimensions: {', '.join(value.shape.spatial.names)}")
        else:
            return value


def to_field(obj):
    if isinstance(obj, SampledField):
        return obj
    if isinstance(obj, Geometry):
        return PointCloud(obj)
    if isinstance(obj, Tensor):
        arbitrary_lines_1d = spatial(obj).rank == 1 and 'vector' in obj.shape
        point_cloud = instance(obj) and 'vector' in obj.shape
        if point_cloud or arbitrary_lines_1d:
            return PointCloud(obj)
        elif spatial(obj):
            return CenteredGrid(obj, 0, bounds=Box(math.const_vec(-0.5, spatial(obj)), wrap(spatial(obj), channel('vector')) - 0.5))
        elif 'vector' in obj.shape:
            return PointCloud(math.expand(obj, instance(points=1)), bounds=Cuboid(obj, half_size=math.const_vec(1e-3, obj.shape['vector'])).box())
        elif instance(obj) and not spatial(obj):
            assert instance(obj).rank == 1, "Bar charts must have only one instance dimension"
            vector = channel(vector=instance(obj).names)
            equal_spacing = math.range_tensor(instance(obj), vector)
            lower = expand(-.5, vector)
            upper = expand(equal_spacing.max + .5, vector)
            return PointCloud(equal_spacing, values=obj, bounds=Box(lower, upper))
            # positions = math.layout(instance(obj).item_names, instance(obj))
            # positions = expand(positions, vector)
            # return PointCloud(positions, values=obj)
    raise ValueError(f"Cannot plot {obj}. Tensors, geometries and fields can be plotted.")


def get_default_limits(f: SampledField) -> Box:
    if f._bounds is not None:
        return f.bounds
    # --- Determine element size ---
    if (f.elements.bounding_half_extent() > 0).any:
        size = 2 * f.elements.bounding_half_extent()
    elif isinstance(f, PointCloud) and f.spatial_rank == 1:
        bounds = f.bounds
        count = non_batch(f).non_dual.non_channel.volume
        return Box(bounds.lower - bounds.size / count / 2, bounds.upper + bounds.size / count / 2)
    # elif instance(f) and f.spatial_rank == 1:
    #     lower = expand(-.5, vector)
    #     upper = expand(equal_spacing.max + .5, vector)
    #     size =
    else:
        size = expand(0, f.elements.shape['vector'])
    if (size == 0).all:
        size = math.const_vec(.1, f.elements.shape['vector'])
    bounds = data_bounds(f.elements.center).largest(channel)
    extended_bounds = Cuboid(bounds.center, bounds.half_size + size * 0.6)
    extended_bounds = Box(math.min(extended_bounds.lower, size.shape.without('vector')), math.max(extended_bounds.upper, size.shape.without('vector')))
    if isinstance(f.elements, Point):
        lower = math.where(extended_bounds.lower * bounds.lower < 0, bounds.lower * .9, extended_bounds.lower)
        upper = math.where(extended_bounds.upper * bounds.upper < 0, bounds.lower * .9, extended_bounds.upper)
        extended_bounds = Box(lower, upper)
    return extended_bounds
