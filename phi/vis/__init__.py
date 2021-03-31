"""
Visualization: plotting, interactive user interfaces.

Use `view()` to show fields or field variables in an interactive user interface.

Use `plot()` to plot fields using Matplotlib.

See the user interface documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html
"""
from ._vis_base import show
from ._viewer import Viewer
from ._matplotlib import plot, animate, plot_scalars
from ._vis import view

__all__ = [key for key in globals().keys() if not key.startswith('_')]

__pdoc__ = {
    'Viewer.action_names': False,
    'Viewer.can_progress': False,
    'Viewer.control_names': False,
    'Viewer.curve_names': False,
    'Viewer.field_names': False,
    'Viewer.get_control': False,
    'Viewer.get_curve': False,
    'Viewer.get_field': False,
    'Viewer.run_action': False,
    'Viewer.set_control_value': False,
}
