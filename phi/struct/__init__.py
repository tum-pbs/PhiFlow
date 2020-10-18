"""
Simplifies working with nested structures of lists, tuples, dicts and objects.

Main functions:

* map
* foreach
* flatten
* zip
* names

"""

from .context import unsafe
from .trait import Trait
from .structdef import definition, variable, constant, derived
from .item_condition import DATA, VARIABLES, CONSTANTS, ALL_ITEMS, ignore
from .struct import Struct, kwargs, to_dict, variables, constants, properties_dict, copy_with, isstruct, equal, VALID, INVALID

# pylint: disable-msg = redefined-builtin
from .functions import flatten, unflatten, names, map, map_item, foreach, zip, Trace, compare, print_differences, shape, staticshape, dtype, any, all
