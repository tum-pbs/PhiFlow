import warnings

from phi.math.backend import NoBackendFound, choose_backend
from ._context import _unsafe, skip_validate
from ._item_condition import ALL_ITEMS, context_item_condition
from ._structdef import Item
from ._struct import copy_with, equal, isstruct, to_dict, Struct, VALID, INVALID, items


def flatten(struct, leaf_condition=None, trace=False, item_condition=None):
    """
    Generates a list of all leaves by recursively iterating over the given struct.

    Args:
      struct: struct or leaf
      leaf_condition: optional) function that determines which structs are treated as leaves. Non-structs are always treated as leaves. (Default value = None)
      trace: If True, returns a list of Trace objects instead of values. (Default value = False)
      item_condition: optional) ItemCondition or boolean function that filters which Items are accumulated. (Default value = None)

    Returns:
      list containing all leaves in the struct hierarchy

    """
    def map_leaf(value):
        result.append(value)
        return value
    result = []
    map(map_leaf, struct, leaf_condition, recursive=True, trace=trace, item_condition=item_condition, content_type=INVALID)
    return result


def unflatten(flat, struct, leaf_condition=None, item_condition=None, content_type=None):
    """
    Undoes a `flatten` operation, restoring the contents of a struct from a list.

    Args:
      flat: list holding the flattened contents of a struct compatible with `struct`
      struct: structure to restore data to
      leaf_condition: optional) function that determines which structs are treated as leaves. Non-structs are always treated as leaves. (Default value = None)
      item_condition: optional) ItemCondition or boolean function that filters which Items are accumulated. (Default value = None)
      content_type: optional) Type key to use for new Structs. Defaults to VALID. Item-specific overrides can be defined by calling Item.override using the content_type as key. Override functions must have the signature (parent_struct, value).

    Returns:
      struct compatible with `struct` holding the values from the `flat` list

    """
    flat = list(flat)
    return map(lambda _: flat.pop(0), struct, leaf_condition=leaf_condition, item_condition=item_condition, content_type=content_type)


def names(struct, leaf_condition=None, full_path=True, basename=None, separator='.'):
    def to_name(trace):
        if not full_path:
            return trace.name if basename is None else basename + separator + trace.name
        else:
            return trace.path(separator) if basename is None else basename + separator + trace.path(separator)
    return map(to_name, struct, leaf_condition, recursive=True, trace=True, content_type=names)


def zip(structs, leaf_condition=None, item_condition=None, zip_parents_if_incompatible=False):
    """
    Builds a single struct containing LeaefZip entries from a list of compatible structs.
    Passing zipped structs to 'map' will call the mapping function with the all leaves at equal positions in the structure.
    
    Example `struct.map(lambda x, y: x+y, struct.zip([{0: 'Hello'}, {0: ' World'}]))` returns `{0: 'Hello World'}`.

    Args:
      structs: iterable collection of structs or leaves
      leaf_condition: optional) function that determines which structs are treated as leaves. Non-structs are always treated as leaves. (Default value = None)
      item_condition: optional) ItemCondition or boolean function that filters which Items are zipped. Excluded items should have the same values among all structs. (Default value = None)
      zip_parents_if_incompatible: If True, suppresses IncompatibleStructs errors if structs with non-matching excluded items are encountered. Instead, these structs are treated as leaves and zipped. (Default value = False)

    Returns:
      Single struct matching the structure of any of the given structs and holding LeafZip objects as leaves for non-excluded items
      :raise IncompatibleStructs: If structs with non-matching excluded items are encountered and zip_parents_if_incompatible=False

    """
    # pylint: disable-msg = redefined-builtin
    assert len(structs) > 0
    first = structs[0]
    if isstruct(first, leaf_condition):
        for struct in structs[1:]:
            if not isstruct(struct):
                if zip_parents_if_incompatible:
                    return LeafZip(structs)
                else:
                    raise IncompatibleStructs('Cannot zip %s and %s because the latter is not a struct.' % (first, struct))
            if set(to_dict(struct, item_condition=item_condition).keys()) != set(to_dict(first, item_condition=item_condition).keys()):
                if zip_parents_if_incompatible:
                    return LeafZip(structs)
                else:
                    raise IncompatibleStructs('Cannot zip %s and %s because keys vary:\n%s\n%s' % (first, struct, to_dict(first, item_condition=item_condition).keys(), to_dict(struct, item_condition=item_condition).keys()))

    if not isstruct(first, leaf_condition):
        return LeafZip(structs)

    dicts = [to_dict(struct, item_condition=item_condition) for struct in structs]
    keys = dicts[0].keys()
    new_dict = {}
    for key in keys:
        values = [d[key] for d in dicts]
        values = zip(values, leaf_condition, item_condition=item_condition, zip_parents_if_incompatible=zip_parents_if_incompatible)
        new_dict[key] = values
    return copy_with(first, new_dict, change_type=zip)


class LeafZip(object):
    """
    Created by struct.zip to replace data.
    When a LeafZip is mapped using 'map', the values are passed as multiple arguments (*args).

    Args:

    Returns:

    """

    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        return self.values[item]

    def __repr__(self):
        return repr(self.values)

    def __str__(self):
        return str(self.values)


class IncompatibleStructs(Exception):
    """Thrown when two or more structs are required to have the same structure but do not, e.g. when trying to zip incompatible structs."""

    def __init__(self, *args):
        Exception.__init__(self, *args)


def map(function, struct, leaf_condition=None, recursive=True, trace=False, item_condition=None, content_type=None):
    """
    Iterates over all items of the struct and maps their values according to the specified function.
    Preserves the hierarchical structure of struct, returning an object of the same type and leaving struct untouched.

    Args:
      function: function mapping from leaf values to new values. If not otherwise specified, the new values will be validated before map returns. If trace=True, Trace objects will be passed instead of values. For zipped structs, multiple values or a Trace containing multiple values is passed to function.
      struct: struct or leaf value
      leaf_condition: optional) function that determines which structs are treated as leaves. Non-structs are always treated as leaves. Leaf structs are not iterated over but directly passed to function. (Default value = None)
      recursive: If True, recursively iterates over all non-leaf sub-structs, passing only leaves to function. Otherwise only iterates over direct items of struct; all sub-structs are treated as leaves. (Default value = True)
      trace: If True, passes a Trace object to function instead of the value. Traces contain additional information. (Default value = False)
      item_condition: optional) ItemCondition or boolean function that filters which Items are iterated over. Excluded items are left untouched. If None, the context item condition is used (data-holding items by default).
      content_type: optional) Type key to use for new Structs. Defaults to VALID. Item-specific overrides can be defined by calling Item.override using the content_type as key. Override functions must have the signature (parent_struct, value).

    Returns:
      object of the same type and hierarchy as struct

    """
    # pylint: disable-msg = redefined-builtin
    if trace is True:
        trace = Trace(struct, None, None)
    if item_condition is None:
        item_condition = context_item_condition
    if content_type is None:
        content_type = VALID
    if not isstruct(struct, leaf_condition):
        if trace is False:
            if isinstance(struct, LeafZip):
                return function(*struct.values)
            else:
                return function(struct)
        else:
            return function(trace)
    else:
        new_values = {}
        if not recursive:
            def leaf_condition(_): return True
        for item in items(struct):
            if item_condition(item):
                old_value = item.get(struct)
                if content_type is not VALID and content_type is not INVALID and item.has_override(content_type):
                    new_value = item.get_override(content_type)(struct, old_value)
                else:
                    new_value = map(function, old_value, leaf_condition, recursive,
                                    Trace(old_value, item.name, trace) if trace is not False else False,
                                    item_condition,
                                    content_type)
                new_values[item.name] = new_value
        return copy_with(struct, new_values, change_type=content_type)


def map_item(item, function, struct, leaf_condition=None, recursive=True, content_type=None):
    assert isinstance(item, Item) or isinstance(item, str)

    def item_condition(item_):
        if isinstance(item, str):
            return item_.name == item
        else:
            return item_.name == item.name
    return map(function, struct, leaf_condition=leaf_condition, recursive=recursive, trace=False, item_condition=item_condition, content_type=content_type)


def foreach(function, *structs, leaf_condition=None, recursive=True, trace=False, item_condition=None):
    if len(structs) == 1:
        map(function, structs[0], leaf_condition, recursive, trace, item_condition, content_type=INVALID)
    else:
        struct = zip(structs, leaf_condition, item_condition)
        map(function, struct, leaf_condition, recursive, trace, item_condition, content_type=INVALID)


class Trace(object):
    """
    Used in struct.map if trace=True.
    Trace objects can be used to reference a specific item of a struct or sub-struct as well as gather information about it.

    Args:

    Returns:

    """

    def __init__(self, value, key, parent_trace):
        self.value = value
        self.key = key
        self.parent = parent_trace

    @property
    def name(self):
        if self.key is None:
            return None
        if isinstance(self.key, str):
            return self.key
        else:
            return str(self.key)

    def path(self, separator='.'):
        if self.parent is not None and self.parent.key is not None:
            return self.parent.path(separator) + separator + self.name
        else:
            return self.name if self.name is not None else ''

    def __repr__(self):
        return "%s = %s" % (self.path(), self.value)

    def find_in(self, base_struct):
        if self.parent is not None and self.parent.key is not None:
            base_struct = self.parent.find_in(base_struct)
        attrs = to_dict(base_struct, item_condition=ALL_ITEMS)
        return attrs[self.key]


def compare(structs, leaf_condition=None, recursive=True, item_condition=ALL_ITEMS):
    if len(structs) <= 1:
        return []
    result = set()

    def check(trace):
        value = trace.value
        for other in structs[1:]:
            try:
                other_value = trace.find_in(other)
                if not equal(value, other_value):
                    result.add(trace)
            except (ValueError, KeyError, TypeError):
                result.add(trace)
    map(check, structs[0], leaf_condition=leaf_condition, recursive=recursive, trace=True, item_condition=item_condition, content_type=INVALID)
    return result


def print_differences(struct1, struct2, level=0):
    if level == 0:
        print('Comparing %s with %s' % (struct1, struct2))
    indent = '  ' * level
    if not isstruct(struct1) or not isstruct(struct2):
        if not equal(struct1, struct2):
            print(indent + 'Values not equal: "%s" and "%s".' % (struct1, struct2))
        return
    items1 = to_dict(struct1)
    items2 = to_dict(struct2)
    tested_keys = []
    for key1 in items1.keys():
        if key1 not in items2:
            print(indent + 'Item "%s" is missing from %s.' % (key1, struct2))
        else:
            if not equal(items1[key1], items2[key1]):
                print('Item "%s" differs between %s and %s.' % (key1, struct1, struct2))
            print_differences(items1[key1], items2[key1], level + 1)
        tested_keys.append(key1)
    for key2 in items2.keys():
        if key2 not in tested_keys:
            print(indent + 'Item "%s" is missing from %s.' % (key2, struct1))


def mappable(leaf_condition=None, recursive=True, item_condition=None, unsafe_context=False, content_type=None):
    if unsafe_context:
        warnings.warn("unsafe_context is deprecated. Use content_type=INVALID instead.")

    def decorator(function):
        def broadcast_function(obj, *args, **kwargs):
            def function_with_args(x): return function(x, *args, **kwargs)
            if unsafe_context:
                with _unsafe():
                    result = map(function_with_args, obj, leaf_condition=leaf_condition, recursive=recursive, item_condition=item_condition, content_type=content_type)
            else:
                result = map(function_with_args, obj, leaf_condition=leaf_condition, recursive=recursive, item_condition=item_condition, content_type=content_type)
            return result
        return broadcast_function
    return decorator


def shape(obj, leaf_condition=None, item_condition=None):
    """
    Maps all values of a struct to their respective dynamic shapes using `math.shape()`.
    To specify custom shapes, add an override with key struct.shape to the Item.

    Args:
      obj: struct or leaf
      leaf_condition: optional) leaf_condition passed to `map` (Default value = None)
      item_condition: optional) item_condition passed to `map` (Default value = None)

    Returns:
      Struct of same type holding shapes instead of data

    """
    def get_shape(obj):
        try:
            return choose_backend(obj).shape(obj)
        except NoBackendFound:
            return ()
    if isinstance(obj, Struct):
        if not skip_validate():
            assert obj.content_type is VALID or obj.content_type is INVALID, "shape can only be accessed on data structs but '%s' has content type '%s'" % (type(obj).__name__, obj.content_type)
    return map(get_shape, obj, leaf_condition=leaf_condition, item_condition=item_condition, content_type=shape)


def staticshape(obj, leaf_condition=None, item_condition=None):
    """
    Maps all values of a struct to their respective static shapes using `math.staticshape()`.
    To specify custom static shapes, add an override with key struct.staticshape to the Item.

    Args:
      obj: struct or leaf
      leaf_condition: optional) leaf_condition passed to `map` (Default value = None)
      item_condition: optional) item_condition passed to `map` (Default value = None)

    Returns:
      Struct of same type holding shapes instead of data

    """
    def get_staticshape(obj):
        try:
            return choose_backend(obj).staticshape(obj)
        except NoBackendFound:
            return ()
    if isinstance(obj, Struct):
        if not skip_validate():
            assert obj.content_type is VALID or obj.content_type is INVALID, "staticshape can only be accessed on data structs but '%s' has content type '%s'" % (type(obj).__name__, obj.content_type)
    return map(get_staticshape, obj, leaf_condition=leaf_condition, item_condition=item_condition, content_type=staticshape)


def dtype(obj, leaf_condition=None, item_condition=None):
    """
    Maps all values of a struct to their respective data types using `math.dtype()`.
    To specify custom dtypes, add an override with key struct.dtype to the Item.

    Args:
      obj: struct or leaf
      leaf_condition: optional) leaf_condition passed to `map` (Default value = None)
      item_condition: optional) item_condition passed to `map` (Default value = None)

    Returns:
      Struct of same type holding data types instead of data

    """
    def get_dtype(obj):
        try:
            return choose_backend(obj).dtype(obj)
        except NoBackendFound:
            return type(obj)
    if isinstance(obj, Struct):
        if not skip_validate():
            assert obj.content_type is VALID or obj.content_type is INVALID, "dtype can only be accessed on data structs but '%s' has content type '%s'" % (type(obj).__name__, obj.content_type)
    return map(get_dtype, obj, leaf_condition=leaf_condition, item_condition=item_condition, content_type=dtype)


def any(condition_struct):
    values = flatten(condition_struct)
    for value in values:
        if value:
            return True
    return False


def all(condition_struct):
    values = flatten(condition_struct)
    for value in values:
        if not value:
            return False
    return True
