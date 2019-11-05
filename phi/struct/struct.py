from copy import copy
import numpy as np
import six, inspect
from .context import skip_validate
from . import collection, typedef


def attr(default=None, dims=None):  # , stack_behavior=collection.object
    def decorator(validate):
        item = typedef.Item(validate.__name__, validate, True, default, dims)
        typedef.register_item_by_function(validate, item)
        return item.property()
    return decorator


def prop(default=None, dims=None):  # , stack_behavior=collection.first
    def decorator(validate):
        item = typedef.Item(validate.__name__, validate, False, default, dims)
        typedef.register_item_by_function(validate, item)
        return item.property()
    return decorator


def kwargs(locals, include_self=True, ignore=()):
    assert 'kwargs' in locals
    locals = locals.copy()
    kwargs = locals['kwargs']
    del locals['kwargs']
    locals.update(kwargs)
    if not include_self and 'self' in locals:
        del locals['self']
    if isinstance(ignore, six.string_types):
        ignore = [ignore]
    for ig in ignore:
        if ig in locals:
            del locals[ig]
    return locals


class Struct(object):

    def __init__(self, **kwargs):
        assert isinstance(self, Struct), 'Struct.__init__() called on %s' % type(self)
        self.__struct__ = typedef.get_type(self.__class__)
        for item in self.__struct__.items:
            if item.name not in kwargs:
                kwargs[item.name] = item.default_value
        self.__set__(**kwargs)
        self.__validate__()

    def copied_with(self, **kwargs):
        duplicate = copy(self)
        duplicate.__set__(**kwargs)
        if not skip_validate():  # double-check since __validate__ could be overridden
            duplicate.__validate__()
        return duplicate

    def __set__(self, **kwargs):
        for name, value in kwargs.items():
            item = self.__struct__.find(name)
            item.set(self, value)
        return self

    def __validate__(self):
        if not skip_validate():
            self.__struct__.validate(self)

    def __attributes__(self, include_properties=False):
        if include_properties:
            return {item.name: item.get(self) for item in self.__struct__.items}
        else:
            return {item.name: item.get(self) for item in self.__struct__.items if item.is_attribute}

    def __properties__(self):
        return {item.name: item.get(self) for item in self.__struct__.items if not item.is_attribute}

    def __properties_dict__(self):
        result = {item.name: properties_dict(getattr(self, item.name))
                  for item in self.__struct__.items if not item.is_attribute}
        for a in self.__struct__.attributes:
            if isstruct(a.get(self)):
                result[a.name] = properties_dict(a.get(self))
        result['type'] = str(self.__class__.__name__)
        result['module'] = str(self.__class__.__module__)
        return result

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        for item in self.__struct__.items:
            if not equal(item.get(self), item.get(other)): return False
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


def attributes(struct, include_properties=False):
    if isinstance(struct, Struct):
        return struct.__attributes__(include_properties=include_properties)
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
        return {'type': str(struct.__class__.__name__), 'module': str(struct.__class__.__module__)}


def copy_with(struct, new_values_dict):
    if isinstance(struct, Struct):
        return struct.copied_with(**new_values_dict)
    if isinstance(struct, tuple):
        duplicate = list(struct)
        for key, value in new_values_dict.items():
            duplicate[key] = value
        return tuple(duplicate)
    if isinstance(struct, list):
        duplicate = list(struct)
        for key, value in new_values_dict.items():
            duplicate[key] = value
        return duplicate
    if isinstance(struct, np.ndarray) and struct.dtype == np.object:
        duplicate = struct.copy()
        for key, value in new_values_dict.items():
            duplicate[key] = value
        return duplicate
    if isinstance(struct, dict):
        duplicate = dict(struct)
        for key, value in new_values_dict.items():
            duplicate[key] = value
        return duplicate
    raise ValueError("Not a struct: %s" % struct)


def isstruct(object, leaf_condition=None):
    isstructclass = isinstance(object, (Struct, list, tuple, dict, np.ndarray))
    if not isstructclass:
        return False
    if isinstance(object, np.ndarray) and object.dtype != np.object:
        return False
    if leaf_condition is not None and leaf_condition(object):
        return False
    return True


def equal(v1, v2):
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