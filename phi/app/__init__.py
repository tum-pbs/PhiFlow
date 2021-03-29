"""
Interactive application development and web interface.

See the user interface documentation at https://tum-pbs.github.io/PhiFlow/Web_Interface.html
"""
from ._value import EditableValue, EditableFloat, EditableInt, EditableBool, EditableString
from ._app import App
from ._display import show, view
from ._plot_util import plot_scalars

__all__ = [key for key in globals().keys() if not key.startswith('_')]
