from .struct import *
from .context import anytype
import six


def flatten(struct, leaf_condition=None, trace=False, include_properties=False):
    list = []
    def map_leaf(value):
        list.append(value)
        return value
    with anytype(): map(map_leaf, struct, leaf_condition, recursive=True, trace=trace, include_properties=include_properties)
    return list


def names(struct, leaf_condition=None, full_path=True, basename=None, separator='.'):
    def f(attr):
        if not full_path:
            return attr.name if basename is None else basename + separator + attr.name
        else:
            return attr.path(separator) if basename is None else basename + separator + attr.path(separator)
    with anytype():
        return map(f, struct, leaf_condition, recursive=True, trace=True)


def zip(structs, leaf_condition=None, include_properties=False, zip_parents_if_incompatible=False):
    assert len(structs) > 0
    first = structs[0]
    if isstruct(first, leaf_condition):
        for s in structs[1:]:
            if attributes(s, include_properties=include_properties).keys() != attributes(first, include_properties=include_properties).keys():
                if zip_parents_if_incompatible:
                    return LeafZip(structs)
                else:
                    raise IncompatibleStructs('Cannot zip %s and %s' % (s, first))

    if not isstruct(first, leaf_condition):
        return LeafZip(structs)

    dicts = [attributes(struct, include_properties=include_properties) for struct in structs]
    keys = dicts[0].keys()
    new_dict = {}
    for key in keys:
        values = [d[key] for d in dicts]
        values = zip(values, leaf_condition, include_properties, zip_parents_if_incompatible)
        new_dict[key] = values
    with anytype():
        return copy_with(first, new_dict)


class LeafZip(object):

    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        return self.values[item]

    def __repr__(self):
        return repr(self.values)

    def __str__(self):
        return str(self.values)


class IncompatibleStructs(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


def map(f, struct, leaf_condition=None, recursive=True, trace=False, include_properties=False):
    if trace is True:
        trace = Trace(struct, None, None)
    if not isstruct(struct, leaf_condition):
        if trace is False:
            if isinstance(struct, LeafZip):
                return f(*struct.values)
            else:
                return f(struct)
        else:
            return f(trace)
    else:
        old_values = attributes(struct, include_properties=include_properties)
        new_values = {}
        if not recursive:
            leaf_condition = lambda x: True
        for key, value in old_values.items():
            new_values[key] = map(f, value, leaf_condition, recursive,
                                  Trace(value, key, trace) if trace is not False else False, include_properties)

        return copy_with(struct, new_values)


class Trace(object):

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
        attrs = attributes(base_struct, include_properties=True)
        return attrs[self.key]


def compare(structs, leaf_condition=None, recursive=True, include_properties=True):
    if len(structs) <= 1: return []
    result = set()
    def check(trace):
        value = trace.value
        for other in structs[1:]:
            try:
                other_value = trace.find_in(other)
                if not equal(value, other_value):
                    result.add(trace)
            except:
                result.add(trace)
    with anytype(): map(check, structs[0], leaf_condition=leaf_condition, recursive=recursive,
                        trace=True, include_properties=include_properties)
    return result