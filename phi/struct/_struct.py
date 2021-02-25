# pylint: disable-msg = redefined-outer-name  # kwargs should be accessed as struct.kwargs
import json
from copy import copy

import numpy as np

from ._context import skip_validate
from ._item_condition import context_item_condition, VARIABLES, CONSTANTS
from ._structdef import Item, derived, _IndexItem


def kwargs(locals, include_self=False, ignore=()):
    # pylint: disable-msg = redefined-builtin
    assert 'kwargs' in locals, "No 'kwargs' variable found in locals. Maybe you forgot to add '**kwargs' as a parameter."
    locals = locals.copy()
    kwargs_in_locals = locals['kwargs']
    del locals['kwargs']
    locals.update(kwargs_in_locals)
    if not include_self and 'self' in locals:
        del locals['self']
    if isinstance(ignore, str):
        ignore = [ignore]
    for ignored_name in ignore:
        if ignored_name in locals:
            del locals[ignored_name]
    return locals


class _DataType(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


INVALID = _DataType('invalid')
VALID = _DataType('valid')


class Struct(object):
    """
    Deprecated.
    
    Base class for all custom structs.
    To implement a custom struct, extend this class and add the decorator @struct.definition().

    Args:

    Returns:

    """

    __items__ = None
    __traits__ = None
    __initialized_class__ = None

    def __init__(self, content_type=VALID, **kwargs):
        assert isinstance(self, Struct), 'Struct.__init__() called on %s. Maybe you forgot **' % type(self)
        assert self.__initialized_class__ == self.__class__, "Instancing %s before struct class is initialized. Maybe you forgot to decorate the class with @struct.definition()" % self.__class__.__name__
        self.__content_type__ = INVALID if content_type is VALID else content_type  # VALID, INVALID, Item for property, string for custom
        for item in self.__items__:
            if item.name not in kwargs:
                kwargs[item.name] = item.default_value
        self._set_items(**kwargs)
        for trait in self.__traits__:
            trait.endow(self)
        if content_type is not INVALID:
            self.validate()

    @derived()
    def shape(self):
        """
        Retrieves the dynamic shapes of items specified through the context (see :class:`phi.struct.item_condition.ItemCondition`).
        Shapes of sub-structs are obtained using `struct.shape` while shapes of non-structs are obtained using `math.shape()`.
        
        To override the shapes of items, use `Item.override` with key `struct.shape` instead of overriding this method.
        
        The result of `x.shape` is equivalent to calling `struct.shape(x)`.
        :return: Struct of same type holding shapes instead of data

        Args:

        Returns:

        """
        from ._struct_functions import shape
        return shape(self)

    @derived()
    def staticshape(self):
        """
        Retrieves the static shapes of items specified through the context (see :class:`phi.struct.item_condition.ItemCondition`).
        Shapes of sub-structs are obtained using `struct.staticshape` while shapes of non-structs are obtained using `math.staticshape()`.
        
        To override the static shapes of items, use `Item.override` with key `struct.staticshape` instead of overriding this method.
        
        The result of `x.staticshape` is equivalent to calling `struct.staticshape(x)`.
        :return: Struct of same type holding shapes instead of data

        Args:

        Returns:

        """
        from ._struct_functions import staticshape
        return staticshape(self)

    @derived()
    def dtype(self):
        """
        Retrieves the data types of items specified through the context (see :class:`phi.struct.item_condition.ItemCondition`).
        Data types of sub-structs are obtained using `struct.dtype` while types of non-structs are obtained using `math.dtype()`.
        
        To override the dtype of items, use `Item.override` with key `struct.dtype` instead of overriding this method.
        
        The result of `x.dtype` is equivalent to calling `struct.dtype(x)`.
        :return: Struct of same type holding data types instead of data

        Args:

        Returns:

        """
        from ._struct_functions import dtype
        return dtype(self)

    def map(self, function, leaf_condition=None, recursive=True, trace=False, item_condition=None, content_type=None):
        """
        Alias for struct.map()

        Args:
          function: 
          leaf_condition:  (Default value = None)
          recursive:  (Default value = True)
          trace:  (Default value = False)
          item_condition:  (Default value = None)
          content_type:  (Default value = None)

        Returns:

        """
        from ._struct_functions import map
        return map(function, self, leaf_condition=leaf_condition, recursive=recursive, trace=trace, item_condition=item_condition, content_type=content_type)

    def map_item(self, item, function, leaf_condition=None, recursive=True, content_type=None):
        """
        Alias for struct.map_item()

        Args:
          item: 
          function: 
          leaf_condition:  (Default value = None)
          recursive:  (Default value = True)
          content_type:  (Default value = None)

        Returns:

        """
        from ._struct_functions import map_item
        return map_item(item, function, self, leaf_condition=leaf_condition, recursive=recursive, content_type=content_type)

    def copied_with(self, change_type=None, **kwargs):
        """
        Returns a copy of this Struct with some items values changed.
        The Struct, this method is invoked on, remains unaltered.
        The returned struct will be validated unless this struct is not valid or the content_type is set to something different than VALID.

        Args:
          change_type: content type of the returned struct (Default value = None)
          kwargs: Items to change, in the form item_name=new_value.
          **kwargs: 

        Returns:
          Altered copy of this object

        """
        duplicate = copy(self)
        duplicate._set_items(**kwargs)  # pylint: disable-msg = protected-access
        target_type = change_type if change_type is not None else self.__content_type__
        if target_type is VALID and not duplicate.is_valid:
            duplicate.__content_type__ = INVALID
            duplicate.validate()
        else:
            duplicate.__content_type__ = target_type
        return duplicate

    def _set_items(self, **kwargs):
        if len(kwargs) == 0:
            return
        if self.is_valid:
            self.__content_type__ = INVALID
        for name, value in kwargs.items():
            try:
                item = getattr(self.__class__, name)
            except (KeyError, TypeError):
                raise TypeError('Struct %s has no property %s' % (self, name))
            item.set(self, value)

    def validate(self):
        """
        Performs validation on this struct if it holds data and is invalid.
        Data-holding structs should always be valid while structs holding non-data content such as shapes or data types are not regarded as valid.
        :return: True if validation was performed, False otherwise

        Args:

        Returns:

        """
        if not skip_validate() and self.__can_validate__():
            self.__validate__()
            if self.__content_type__ is INVALID:
                self.__content_type__ = VALID
            return True
        else:
            return False

    def __can_validate__(self):
        return self.__content_type__ is INVALID

    def __validate__(self):
        for trait in self.__traits__:
            trait.pre_validate_struct(self)
        for item in self.__items__:
            item.validate(self)
        for trait in self.__traits__:
            trait.post_validate_struct(self)

    @property
    def is_valid(self):
        return self.__content_type__ is VALID

    @property
    def content_type(self):
        return self.__content_type__

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

    def __repr__(self):
        return "%s[%s]" % (type(self).__name__, self.content_type)


def to_dict(struct, item_condition=None):
    if item_condition is None:
        item_condition = context_item_condition
    if isinstance(struct, Struct):
        return struct.__to_dict__(item_condition)
    if isinstance(struct, (list, tuple, np.ndarray)):
        return {i: struct[i] for i in range(len(struct)) if item_condition(Item(name=i, validation_function=None, is_variable=True, default_value=None, dependencies=(), holds_data=True))}
    if isinstance(struct, dict):
        return struct
    raise ValueError("Not a struct: %s" % struct)


def items(struct):
    if isinstance(struct, Struct):
        return struct.__items__
    if isinstance(struct, (list, tuple, np.ndarray)):
        return [_IndexItem(i) for i in range(len(struct))]
    if isinstance(struct, dict):
        return [_IndexItem(key) for key in struct.keys()]
    raise ValueError("Not a struct: '%s'" % struct)


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


def copy_with(struct, new_values_dict, change_type=None):
    if isinstance(struct, Struct):
        return struct.copied_with(change_type=change_type, **new_values_dict)
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
        if type(struct) is dict:
            return duplicate
        else:
            return type(struct)(duplicate)
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
