"""
Interactive application development and web interface.

See the user interface documentation at https://github.com/tum-pbs/PhiFlow/blob/develop/documentation/Web_Interface.md
"""
from ._value import EditableValue, EditableFloat, EditableInt, EditableBool, EditableString
from ._app import App
from ._display import show

__all__ = [key for key in globals().keys() if not key.startswith('_')]
