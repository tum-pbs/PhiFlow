import six

from .context import unsafe
from .item_condition import ALL_ITEMS, context_item_condition
from .struct import copy_with, equal, isstruct, to_dict


def flatten(struct, leaf_condition=None, trace=False, item_condition=None):
    """
Generates a list of all leaves by recursively iterating over the given struct.
    :param struct: struct or leaf
    :param leaf_condition: (optional) function that determines which structs are treated as leaves. Non-structs are always treated as leaves.
    :param trace: If True, returns a list of Trace objects instead of values.
    :param item_condition: (optional) ItemCondition or boolean function that filters which Items are accumulated.
    :return: list containing all leaves in the struct hierarchy
    """
    def map_leaf(value):
        result.append(value)
        return value
    result = []
    with unsafe():
        map(map_leaf, struct, leaf_condition, recursive=True, trace=trace, item_condition=item_condition)
    return result


def names(struct, leaf_condition=None, full_path=True, basename=None, separator='.'):
    def to_name(trace):
        if not full_path:
            return trace.name if basename is None else basename + separator + trace.name
        else:
            return trace.path(separator) if basename is None else basename + separator + trace.path(separator)
    with unsafe():
        return map(to_name, struct, leaf_condition, recursive=True, trace=True)


def zip(structs, leaf_condition=None, item_condition=None, zip_parents_if_incompatible=False):
    """
Builds a single struct containing LeaefZip entries from a list of compatible structs.
Passing zipped structs to 'map' will call the mapping function with the all leaves at equal positions in the structure.

Example `struct.map(lambda x, y: x+y, struct.zip([{0: 'Hello'}, {0: ' World'}]))` returns `{0: 'Hello World'}`.
    :param structs: iterable collection of structs or leaves
    :param leaf_condition: (optional) function that determines which structs are treated as leaves. Non-structs are always treated as leaves.
    :param item_condition: (optional) ItemCondition or boolean function that filters which Items are zipped. Excluded items should have the same values among all structs.
    :param zip_parents_if_incompatible: If True, suppresses IncompatibleStructs errors if structs with non-matching excluded items are encountered. Instead, these structs are treated as leaves and zipped.
    :return: Single struct matching the structure of any of the given structs and holding LeafZip objects as leaves for non-excluded items
    :raise IncompatibleStructs: If structs with non-matching excluded items are encountered and zip_parents_if_incompatible=False
    """
    # pylint: disable-msg = redefined-builtin
    assert len(structs) > 0
    first = structs[0]
    if isstruct(first, leaf_condition):
        for struct in structs[1:]:
            if set(to_dict(struct, item_condition=item_condition).keys()) != set(to_dict(first, item_condition=item_condition).keys()):
                if zip_parents_if_incompatible:
                    return LeafZip(structs)
                else:
                    raise IncompatibleStructs('Cannot zip %s and %s because keys vary:\n%s\n%s' % (struct, first, to_dict(struct, item_condition=item_condition).keys(), to_dict(first, item_condition=item_condition).keys()))

    if not isstruct(first, leaf_condition):
        return LeafZip(structs)

    dicts = [to_dict(struct, item_condition=item_condition) for struct in structs]
    keys = dicts[0].keys()
    new_dict = {}
    for key in keys:
        values = [d[key] for d in dicts]
        values = zip(values, leaf_condition, item_condition=item_condition, zip_parents_if_incompatible=zip_parents_if_incompatible)
        new_dict[key] = values
    with unsafe():
        return copy_with(first, new_dict)


class LeafZip(object):
    """
Created by struct.zip to replace data.
When a LeafZip is mapped using 'map', the values are passed as multiple arguments (*args).
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
    """
Thrown when two or more structs are required to have the same structure but do not, e.g. when trying to zip incompatible structs.
    """
    def __init__(self, *args):
        Exception.__init__(self, *args)


def map(function, struct, leaf_condition=None, recursive=True, trace=False, item_condition=None):
    """
Iterates over all items of the struct and maps their values according to the specified function.
Preserves the hierarchical structure of struct, returning an object of the same type and leaving struct untouched.
    :param function: function mapping from leaf values to new values. If not otherwise specified, the new values will be validated before map returns. If trace=True, Trace objects will be passed instead of values. For zipped structs, multiple values or a Trace containing multiple values is passed to function.
    :param struct: struct or leaf value
    :param leaf_condition: (optional) function that determines which structs are treated as leaves. Non-structs are always treated as leaves. Leaf structs are not iterated over but directly passed to function.
    :param recursive: If True, recursively iterates over all non-leaf sub-structs, passing only leaves to function. Otherwise only iterates over direct items of struct; all sub-structs are treated as leaves.
    :param trace: If True, passes a Trace object to function instead of the value. Traces contain additional information.
    :param item_condition: (optional) ItemCondition or boolean function that filters which Items are iterated over. Excluded items are left untouched. If None, the context item condition is used (data-holding items by default).
    :return: object of the same type and hierarchy as struct
    """
    # pylint: disable-msg = redefined-builtin
    if trace is True:
        trace = Trace(struct, None, None)
    if item_condition is None:
        item_condition = context_item_condition
    if not isstruct(struct, leaf_condition):
        if trace is False:
            if isinstance(struct, LeafZip):
                return function(*struct.values)
            else:
                return function(struct)
        else:
            return function(trace)
    else:
        old_values = to_dict(struct, item_condition=item_condition)
        new_values = {}
        if not recursive:
            leaf_condition = lambda x: True
        for key, value in old_values.items():
            new_values[key] = map(function, value, leaf_condition, recursive,
                                  Trace(value, key, trace) if trace is not False else False,
                                  item_condition=item_condition)

        return copy_with(struct, new_values)


class Trace(object):
    """
Used in struct.map if trace=True.
Trace objects can be used to reference a specific item of a struct or sub-struct as well as gather information about it.
    """

    def __init__(self, value, key, parent_trace):
        self.value = value
        self.key = key
        self.parent = parent_trace

    @property
    def name(self):
        if self.key is None:
            return None
        if isinstance(self.key, six.string_types):
            return self.key
        else:
            return str(self.key)

    def path(self, separator='.'):
        if self.parent is not None and self.parent.key is not None:
            return self.parent.path(separator) + separator + self.name
        else:
            return self.name

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
    with unsafe():
        map(check, structs[0], leaf_condition=leaf_condition, recursive=recursive, trace=True, item_condition=item_condition)
    return result


def print_differences(struct1, struct2, level=0):
    if level == 0:
        print('Comparing %s with %s' % (struct1, struct2))
    indent = '  ' * level
    if not isstruct(struct1) or not isstruct(struct2):
        if not equal(struct1, struct2):
            print(indent+'Values not equal: "%s" and "%s".' % (struct1, struct2))
        return
    items1 = to_dict(struct1)
    items2 = to_dict(struct2)
    tested_keys = []
    for key1 in items1.keys():
        if key1 not in items2:
            print(indent+'Item "%s" is missing from %s.' % (key1, struct2))
        else:
            if not equal(items1[key1], items2[key1]):
                print('Item "%s" differs between %s and %s.' % (key1, struct1, struct2))
            print_differences(items1[key1], items2[key1], level+1)
        tested_keys.append(key1)
    for key2 in items2.keys():
        if key2 not in tested_keys:
            print(indent+'Item "%s" is missing from %s.' % (key2, struct1))


def mappable(leaf_condition=None, recursive=True, item_condition=None, unsafe_context=False):
    def decorator(function):
        def broadcast_function(obj, *args, **kwargs):
            def function_with_args(x): return function(x, *args, **kwargs)
            if unsafe_context:
                with unsafe():
                    result = map(function_with_args, obj, leaf_condition=leaf_condition, recursive=recursive, item_condition=item_condition)
            else:
                result = map(function_with_args, obj, leaf_condition=leaf_condition, recursive=recursive, item_condition=item_condition)
            return result
        return broadcast_function
    return decorator
