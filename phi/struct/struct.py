# pylint: disable-msg = redefined-outer-name  # kwargs should be accessed as struct.kwargs
import json
from copy import copy

import numpy as np
import six

from ..backend.dynamic_backend import DYNAMIC_BACKEND as math
from .context import skip_validate
from .item_condition import context_item_condition, VARIABLES, CONSTANTS
from .structdef import Item, derived


def kwargs(locals, include_self=False, ignore=()):
    # pylint: disable-msg = redefined-builtin
    assert 'kwargs' in locals, "No 'kwargs' variable found in locals. Maybe you forgot to add '**kwargs' as a parameter."
    locals = locals.copy()
    kwargs_in_locals = locals['kwargs']
    del locals['kwargs']
    locals.update(kwargs_in_locals)
    if not include_self and 'self' in locals:
        del locals['self']
    if isinstance(ignore, six.string_types):
        ignore = [ignore]
    for ignored_name in ignore:
        if ignored_name in locals:
            del locals[ignored_name]
    return locals


class Struct(object):
    """
Base class for all custom structs.
To implement a custom struct, extend this class and add the decorator @struct.definition().

See the struct documentation at documentation/Structs.ipynb
    """

    __items__ = None
    __traits__ = None
    __initialized_class__ = None

    def __init__(self, **kwargs):
        assert isinstance(self, Struct), 'Struct.__init__() called on %s. Maybe you forgot **' % type(self)
        assert self.__initialized_class__ == self.__class__, "Instancing %s before struct class is initialized. Maybe you forgot to decorate the class with @struct.definition()" % self.__class__.__name__
        for item in self.__items__:
            if item.name not in kwargs:
                kwargs[item.name] = item.default_value
        self._set_items(**kwargs)
        for trait in self.__traits__:
            trait.endow(self)
        self.validate()

    @derived()
    def shape(self):
        """
Maps all DATA values to their respective dynamic shapes.
Shapes of sub-structs are obtained using struct.shape while shapes of non-structs are obtained using math.shape().
Struct subclasses can override this method e.g. to specify unknown dimensions (although the current data has a known dimension).
        :return: Invalid struct holding shapes instead of data
        """
        duplicate = copy(self)
        for item in duplicate.__items__:
            if context_item_condition(item):
                obj = item.get(duplicate)
                if item.has_override(Struct.shape):
                    shape = item.get_override(Struct.shape)(duplicate, obj)
                else:
                    shape = Struct.shape(obj) if isstruct(obj) else math.shape(obj)
                item.set(duplicate, shape)
        return duplicate

    @derived()
    def staticshape(self):
        """
Maps all DATA values to their respective static shapes.
Shapes of sub-structs are obtained using struct.staticshape while shapes of non-structs are obtained using math.staticshape().
Struct subclasses can override this method e.g. to specify unknown dimensions (although the current data has a known dimension).
        :return: Invalid struct holding shapes instead of data
        """
        duplicate = copy(self)
        for item in duplicate.__items__:
            if context_item_condition(item):
                obj = item.get(duplicate)
                if item.has_override(Struct.staticshape):
                    shape = item.get_override(Struct.staticshape)(duplicate, obj)
                else:
                    shape = Struct.staticshape(obj) if isstruct(obj) else math.staticshape(obj)
                item.set(duplicate, shape)
        return duplicate

    def copied_with(self, **kwargs):
        """
Returns a copy of this Struct with some items values changed.
The Struct, this method is invoked on, remains unaltered.
Unless otherwise specified, the returned object will be validated, i.e. the new item values may be altered before the new object is returned.
        :param kwargs: Items to change, in the form item_name=new_value.
        :return: Altered copy of this object
        """
        duplicate = copy(self)
        duplicate._set_items(**kwargs)  # pylint: disable-msg = protected-access
        duplicate.validate()
        return duplicate

    def _set_items(self, **kwargs):
        for name, value in kwargs.items():
            try:
                item = getattr(self.__class__, name)
            except (KeyError, TypeError):
                raise TypeError('Struct %s has no property %s' % (self, name))
            item.set(self, value)
        return self

    def validate(self):
        """
Performs validation on this struct.
Structs are always valid unless otherwise specified.
A user need only invoke this method when explicitly dealing with invalid structs.
        """
        if not skip_validate():
            self.__validate__()

    def __validate__(self):
        for trait in self.__traits__:
            trait.pre_validate_struct(self)
        for item in self.__items__:
            item.validate(self)
        for trait in self.__traits__:
            trait.post_validate_struct(self)

    def __to_dict__(self, item_condition):
        if item_condition is not None:
            return {item.name: item.get(self) for item in self.__items__ if item_condition(item)}
        else:
            return {item.name: item.get(self) for item in self.__items__}

    def __properties_dict__(self):
        result = {item.name: properties_dict(getattr(self, item.name)) for item in self.__items__ if not item.holds_data}
        for item in self.__items__:
            if isstruct(item.get(self)):
                result[item.name] = properties_dict(item.get(self))
        result['type'] = str(self.__class__.__name__)
        result['module'] = str(self.__class__.__module__)
        return result

    def __eq__(self, other):
        if type(self) != type(other):  # pylint: disable-msg = unidiomatic-typecheck
            return False
        for item in self.__items__:
            if not equal(item.get(self), item.get(other)):
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        hash_value = 0
        for attr in self.__to_dict__(None).values():
            try:
                hash_value += hash(attr)
            except TypeError:  # unhashable type
                pass
        return hash_value


def to_dict(struct, item_condition=None):
    if isinstance(struct, Struct):
        return struct.__to_dict__(item_condition)
    if isinstance(struct, (list, tuple, np.ndarray)):
        if item_condition is None:
            return {i: struct[i] for i in range(len(struct))}
        else:
            return {i: struct[i] for i in range(len(struct)) if item_condition(Item(name=i, validation_function=None, is_variable=True, default_value=None, dependencies=(), holds_data=True))}
    if isinstance(struct, dict):
        return struct
    raise ValueError("Not a struct: %s" % struct)


def variables(struct):
    return to_dict(struct, VARIABLES)


def constants(struct):
    if isinstance(struct, (list, tuple, dict, np.ndarray)):
        return {}
    else:
        return to_dict(struct, CONSTANTS)


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
    try:
        json.dumps(struct)
        return struct
    except TypeError:  # not serializable
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


def isstruct(obj, leaf_condition=None):
    if not isinstance(obj, (Struct, list, tuple, dict, np.ndarray)):
        return False
    if isinstance(obj, np.ndarray) and obj.dtype != np.object:
        return False
    if leaf_condition is not None and leaf_condition(obj):
        return False
    return True


def equal(obj1, obj2):
    if isinstance(obj1, np.ndarray) or isinstance(obj2, np.ndarray):
        if obj1.dtype != np.object and obj2.dtype != np.object:
            if not np.allclose(obj1, obj2):
                return False
        else:
            if not np.all(obj1 == obj2):
                return False
    else:
        try:
            if obj1 != obj2:
                return False
            # pylint: disable-msg = broad-except  # the exception type can depend on the values
        except (ValueError, BaseException):  # not a boolean result
            if obj1 is not obj2:
                return False
    return True
