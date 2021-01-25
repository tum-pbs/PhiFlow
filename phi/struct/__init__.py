"""
*Deprecated*.

Simplifies working with nested structures of lists, tuples, dicts and objects.

Main functions:

* map
* foreach
* flatten
* zip
* names

"""

from ._context import unsafe
from ._trait import Trait
from ._structdef import definition, variable, constant, derived
from ._item_condition import DATA, VARIABLES, CONSTANTS, ALL_ITEMS, ignore
from ._struct import Struct, kwargs, to_dict, variables, constants, properties_dict, copy_with, isstruct, equal, VALID, INVALID

# pylint: disable-msg = redefined-builtin
from ._struct_functions import flatten, unflatten, names, map, map_item, foreach, zip, Trace, compare, print_differences, shape, staticshape, dtype, any, all


__all__ = [key for key in globals().keys() if not key.startswith('_')]
