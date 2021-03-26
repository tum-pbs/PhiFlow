"""
Interactive application development and web interface.

See the user interface documentation at https://tum-pbs.github.io/PhiFlow/Web_Interface.html
"""
from ._value import EditableValue, EditableFloat, EditableInt, EditableBool, EditableString
from ._app import App
from ._module_app import ModuleViewer
from ._jupyter import NotebookViewer
from ._display import show
from ._plot_util import plot_scalars

__all__ = [key for key in globals().keys() if not key.startswith('_')]
