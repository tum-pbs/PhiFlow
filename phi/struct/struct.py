from copy import copy
import numpy as np
from .context import skip_validate


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
        duplicate._set(**kwargs)
        if not skip_validate():  # double-check since __validate__ could be overridden
            duplicate.__validate__(kwargs.keys())
        return duplicate

    def _set(self, **kwargs):
        for name, value in kwargs.items():
            attr = self.__class__.__struct__.find(name).ref_string
            try:
                setattr(self, attr, value)
            except AttributeError as e:
                raise AttributeError("can't copy struct %s because attribute %s cannot be set." % (self, attr))
        return self

    def __validate__(self, attribute_names=None):
        if skip_validate(): return
        if attribute_names is None:
            attribute_names = [a.name for a in self.__class__.__struct__.all]
        for name in attribute_names:
            validate_name = '__validate_%s__' % name
            if hasattr(self, validate_name):
                getattr(self, validate_name)()

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
        if type(self) != type(other):
            return False
        for attr in self.__class__.__struct__.all:
            v1 = getattr(self, attr.name)
            v2 = getattr(other, attr.name)
            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if v1.dtype != np.object and v2.dtype != np.object:
                    if not np.allclose(v1, v2):
                        return False
                else:
                    if not np.all(v1 == v2):
                        return False
            else:
                try:
                    if v1 != v2:
                        return False
                except:
                    if v1 is not v2:
                        return False
        return True

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        h = 0
        for attr in self.__attributes__().values():
            try:
                h += hash(attr)
            except:
                pass
        return h


def attributes(struct):
    if isinstance(struct, Struct):
        return struct.__attributes__()
    if isinstance(struct, (list, tuple, np.ndarray)):
        return {i: struct[i] for i in range(len(struct))}
    if isinstance(struct, dict):
        return struct
    raise ValueError("Not a struct: %s" % struct)


def properties(struct):
    if isinstance(struct, Struct):
        return struct.__properties__()
    if isinstance(struct, (list, tuple, dict, np.ndarray)):
        return {}
    raise ValueError("Not a struct: %s" % struct)


def properties_dict(struct):
    if isinstance(struct, Struct):
        return struct.__properties_dict__()
    if isinstance(struct, (list, tuple)):
        return [properties_dict(s) for s in struct]
    if isinstance(struct, np.ndarray) and struct.dtype == np.object:
        return [properties_dict(s) for s in struct]
    if isinstance(struct, dict):
        return {key: properties_dict(value) for key,value in struct.items()}
    if isinstance(struct, np.ndarray):
        struct = struct.tolist()
    import json
    try:
        json.dumps(struct)
        return struct
    except:
        raise TypeError('Object "%s" of type %s is not JSON serializable' % (struct,type(struct)))


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
    if isinstance(struct, np.ndarray) and struct.dtype == np.object:
        duplicate = struct.copy()
        for key, value in new_values_dict.items(): duplicate[key] = value
        return duplicate
    if isinstance(struct, dict):
        duplicate = dict(struct)
        for key, value in new_values_dict.items(): duplicate[key] = value
        return duplicate
    raise ValueError("Not a struct: %s" % struct)


def isstruct(object, leaf_condition=None):
    isstructclass =  isinstance(object, (Struct, list, tuple, dict, np.ndarray))
    if not isstructclass:
        return False
    if isinstance(object, np.ndarray) and object.dtype != np.object:
        return False
    if leaf_condition is not None and leaf_condition(object):
        return False
    return True


