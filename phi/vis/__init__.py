"""
Visualization: plotting, interactive user interfaces.

See the user interface documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html
"""
from ._value import EditableValue, EditableFloat, EditableInt, EditableBool, EditableString
from ._viewer import Viewer
from ._display import show, view
from ._matplotlib import plot, animate, plot_scalars

__all__ = [key for key in globals().keys() if not key.startswith('_')]
