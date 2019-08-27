from copy import copy
import numpy as np
import six


class StructAttr(object):

    def __init__(self, ref_string, name=None, is_property=False):
        self.ref_string = ref_string
        self.name = name if name is not None else (ref_string[1:] if ref_string.startswith('_') else ref_string)
        self.is_property = is_property

    def matches(self, string):
        return self.name == string or self.ref_string == string

    def __repr__(self):
        return self.ref_string


class Def(object):

    def __init__(self, attributes, properties=()):
        assert isinstance(attributes, (list, tuple))
        assert isinstance(properties, (list, tuple))
        self.attributes = [a if isinstance(a, StructAttr) else StructAttr(a, is_property=False) for a in attributes]
        self.properties = [p if isinstance(p, StructAttr) else StructAttr(p, is_property=True) for p in properties]
        self.all = self.attributes + self.properties

    def find_attribute(self, name):
        for structattribute in self.attributes:
            if structattribute.matches(name):
                return structattribute
        raise KeyError('No attribute %s' % name)

    def find(self, name):
        for structattribute in self.all:
            if structattribute.matches(name):
                return structattribute
        raise KeyError('No attribute or property: %s' % name)

    def extend(self, attributes, properties=()):
        return Def(self.attributes + list(attributes), self.properties + list(properties))


class Struct(object):
    __struct__ = Def(())

    def copied_with(self, **kwargs):
        duplicate = copy(self)
        for name, value in kwargs.items():
            attr = self.__class__.__struct__.find(name).ref_string
            setattr(duplicate, attr, value)
        return duplicate

    def __attributes__(self):
        return {a.name: getattr(self, a.name) for a in self.__class__.__struct__.attributes}

    def __properties__(self):
        return {p.name: getattr(self, p.name) for p in self.__class__.__struct__.properties}

    def __properties_dict__(self):
        result = {p.name: properties_dict(getattr(self, p.name)) for p in self.__class__.__struct__.properties if p.is_property}
        for a in self.__class__.__struct__.attributes:
            if isstruct(getattr(self, a.name)):
                result[a.name] = properties_dict(getattr(self, a.name))
        result['type'] = str(self.__class__.__name__)
        result['module'] = str(self.__class__.__module__)
        return result

    def __eq__(self, other):
        if type(self) != type(other): return False
        for attr in self.__class__.__struct__.all:
            v1 = getattr(self, attr.name)
            v2 = getattr(other, attr.name)
            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if not np.all(np.equal(v1, v2)): return False
            else:
                try:
                    if v1 != v2: return False
                except:
                    if v1 is not v2: return False
        return True

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(self.__attributes__().values()))


def attributes(struct):
    if isinstance(struct, Struct):
        return struct.__attributes__()
    if isinstance(struct, (list, tuple)):
        return {i: struct[i] for i in range(len(struct))}
    if isinstance(struct, dict):
        return struct
    raise ValueError("Not a struct: %s" % struct)


def properties(struct):
    if isinstance(struct, Struct):
        return struct.__properties__()
    if isinstance(struct, (list, tuple, dict)):
        return {}
    raise ValueError("Not a struct: %s" % struct)


def properties_dict(struct):
    if isinstance(struct, Struct):
        return struct.__properties_dict__()
    if isinstance(struct, (list, tuple)):
        return [properties_dict(s) for s in struct]
    if isinstance(struct, dict):
        return {key: properties_dict(value) for key,value in struct.items()}
    if isinstance(struct, np.ndarray):
        assert len(struct.shape) == 1
        struct = struct.tolist()
    import json
    try:
        json.dumps(struct)
        return struct
    except:
        raise TypeError('Object "%s" of type "%s" is not JSON serializable' % (struct,type(struct)))


def copy_with(struct, new_values_dict):
    if isinstance(struct, Struct):
        return struct.copied_with(**new_values_dict)
    if isinstance(struct, tuple):
        duplicate = list(struct)
        for key, value in new_values_dict.items(): duplicate[key] = value
        return tuple(duplicate)
    if isinstance(struct, list):
        duplicate = list(struct)
        for key, value in new_values_dict.items(): duplicate[key] = value
        return duplicate
    if isinstance(struct, dict):
        duplicate = dict(struct)
        for key, value in new_values_dict.items(): duplicate[key] = value
        return duplicate
    raise ValueError("Not a struct: %s" % struct)


class Trace(object):

    def __init__(self, value, key, parent):
        self.value = value
        self.key = key
        self.parent = parent  # AttributeIdentifier or struct

    @property
    def name(self):
        if self.key is None:
            return None
        if isinstance(self.key, six.string_types):
            return self.key
        else:
            return str(self.key)

    def path(self, separator='.'):
        if isinstance(self.parent, Trace) and self.parent.key is not None:
            return self.parent.path(separator) + separator + self.name
        else:
            return self.name

    def __repr__(self):
        return "%s = %s" % (self.path(), self.value)


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
        old_values = attributes(struct)
        if include_properties:
            for k, v in properties(struct).items():  # TODO properties can return lists
                old_values[k] = v
        new_values = {}
        if not recursive:
            leaf_condition = lambda x: True
        for key, value in old_values.items():
            new_values[key] = map(f, value, leaf_condition, recursive,
                                  Trace(value, key, trace) if trace is not False else False, include_properties)

        return copy_with(struct, new_values)


def zip(structs, leaf_condition=None):
    assert len(structs) > 0
    first = structs[0]
    if isstruct(first, leaf_condition):
        for s in structs[1:]:
            assert type(s) == type(first)

    if not isstruct(first, leaf_condition):
        return LeafZip(structs)

    dicts = [attributes(struct) for struct in structs]
    keys = dicts[0].keys()
    new_dict = {}
    for key in keys:
        values = [d[key] for d in dicts]
        values = zip(values, leaf_condition)
        new_dict[key] = values
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



def isstruct(object, leaf_condition=None):
    isstructclass =  isinstance(object, (Struct, list, tuple, dict))
    if not isstructclass:
        return False
    if leaf_condition is not None and leaf_condition(object):
        return False
    return True


def flatten(struct, leaf_condition=None, trace=False, include_properties=False):
    list = []
    def map_leaf(value):
        list.append(value)
        return value
    map(map_leaf, struct, leaf_condition, recursive=True, trace=trace, include_properties=include_properties)
    return list


def names(struct, leaf_condition=None, full_path=True, basename=None, separator='.'):
    def f(attr):
        if not full_path:
            return attr.name if basename is None else basename + separator + attr.name
        else:
            return attr.path(separator) if basename is None else basename + separator + attr.path(separator)
    return map(f, struct, leaf_condition, recursive=True, trace=True)
