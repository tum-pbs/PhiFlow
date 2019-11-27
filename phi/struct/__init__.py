from .context import anytype
from .structdef import definition, attr, prop
from .struct import Struct, kwargs, attributes, properties, properties_dict, copy_with, isstruct, equal

# pylint: disable-msg = redefined-builtin
from .functions import flatten, names, map, zip, Trace, compare
