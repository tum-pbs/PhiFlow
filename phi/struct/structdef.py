import typing  # pylint: disable-msg = unused-import  # this is used in # type
from typing import Dict
import six


STRUCT_CLASSES = None  # type: tuple


_STRUCT_REGISTER = {}  # type: Dict[typing.Type, StructType]
_UNUSED_ITEMS = {}  # type: Dict[str, Item] # only temporary, before class decorator called


def definition():
    """
Required decorator for custom struct classes.
    """
    def decorator(cls):
        structtype = _build_type(cls)
        cls.__struct__ = structtype
        return cls
    return decorator


def variable(default=None, dependencies=(), holds_data=True):
    """
Required decorator for data_dict of custom structs.
The enclosing class must be decorated with struct.definition().
    :param default: default value passed to validation if no other value is specified
    :param dependencies: other items (string or reference for inherited constants_dict)
    :param holds_data: determines whether the variable is considered by data-related functions
    :return: read-only property
    """
    def decorator(validate):
        item = Item(validate.__name__, validate, True, default, dependencies, holds_data)
        _register_item(validate, item)
        return item
    return decorator


def constant(default=None, dependencies=(), holds_data=False):
    """
Required decorator for constants_dict of custom structs.
The enclosing class must be decorated with struct.definition().
    :param default: default value passed to validation if no other value is specified
    :param dependencies: other items (string or reference for inherited constants_dict)
    :param holds_data: determines whether the constant is considered by data-related functions
    :return: read-only property
    """
    def decorator(validate):
        item = Item(validate.__name__, validate, False, default, dependencies, holds_data)
        _register_item(validate, item)
        return item
    return decorator


def derived():
    """
Derived properties work similar to @property but can be easily broadcast across many instances.
    :return: read-only property
    """
    def decorator(getter):
        return DerivedProperty(getter.__name__, getter)
    return decorator


def _register_item(_function, item):
    _UNUSED_ITEMS[item.name] = item


def _build_type(cls):
    assert cls not in _STRUCT_REGISTER
    items = {}
    for attribute in dir(cls):
        if attribute in _UNUSED_ITEMS:
            items[attribute] = _UNUSED_ITEMS.pop(attribute)
    for base in cls.__bases__:
        if base not in STRUCT_CLASSES and base in _STRUCT_REGISTER:
            basetype = _STRUCT_REGISTER[base]
            for item in basetype.items:
                if item.name not in items:
                    items[item.name] = item
    structtype = StructType(cls, items)
    _STRUCT_REGISTER[cls] = structtype
    return structtype


def get_type(struct_class):
    """
    :rtype: StructType
    """
    return _STRUCT_REGISTER[struct_class]


class StructType(object):
    """
One StructType is associated with each defined struct (subclass of Struct) and stored in the _STRUCT_REGISTER.
    """

    def __init__(self, struct_class, item_dict):
        self.struct_class = struct_class
        self.item_dict = item_dict
        self.items = _order_by_dependencies(item_dict, self)
        self.variables = tuple(filter(lambda item: item.is_variable, self.items))
        self.constants = tuple(filter(lambda item: not item.is_variable, self.items))

    def find(self, name):
        return self.item_dict[name]

    def validate(self, struct):
        for item in self.items:
            item.validate(struct)

    @property
    def item_names(self):
        return [item.name for item in self.items]

    def __repr__(self):
        return self.struct_class.__name__


class Item(object):
    """
Represents an item type of a struct, a variable or a constant.
    """

    def __init__(self, name, validation_function, is_variable, default_value, dependencies, holds_data):
        assert callable(validation_function) or validation_function is None
        self.name = name
        self.validation_function = validation_function
        self.is_variable = is_variable
        self.default_value = default_value
        self.dependencies = dependencies
        self.holds_data = holds_data
        self.owner = None

    def set(self, struct, value):
        try:
            setattr(struct, '_' + self.name, value)
        except AttributeError:
            raise AttributeError("can't modify struct %s because item %s cannot be set." % (struct, self))

    def get(self, struct):
        return getattr(struct, '_' + self.name)

    def validate(self, struct):
        if self.validation_function is not None:
            old_val = self.get(struct)
            new_val = self.validation_function(struct, old_val)
            self.set(struct, new_val)

    def __get__(self, instance, owner):
        if instance is not None:
            return getattr(instance, '_'+self.name)
        else:
            self.owner = owner
            return self

    def __call__(self, obj):
        assert self.owner is not None
        from .functions import map
        return map(lambda x: getattr(x, '_'+self.name), obj, leaf_condition=lambda x: isinstance(x, self.owner))

    def __set__(self, instance, value):
        raise AttributeError('Struct variables and constants are read-only.')

    def __delete__(self, instance):
        raise AttributeError('Struct variables and constants are read-only.')

    def __repr__(self):
        return self.name


def CONSTANTS(item): return not item.is_variable
def VARIABLES(item): return item.is_variable
def DATA(item): return item.holds_data
ALL_ITEMS = None


class DerivedProperty(object):

    def __init__(self, name, getter):
        self.name = name
        self.getter = getter

    def __get__(self, instance, owner):
        if instance is not None:
            return self.getter(instance)
        else:
            self.owner = owner
            return self

    def __call__(self, obj):
        assert self.owner is not None
        from .functions import map
        return map(lambda x: self.getter(x), obj, leaf_condition=lambda x: isinstance(x, self.owner))

    def __set__(self, instance, value):
        raise AttributeError('Derived constants_dict cannot be set.')

    def __delete__(self, instance):
        raise AttributeError('Derived constants_dict cannot be deleted.')

    def __repr__(self):
        return self.name


def _order_by_dependencies(item_dict, owner):
    result = []
    for item in item_dict.values():
        _recursive_deps_add(item, item_dict, result, owner)
    return result


def _recursive_deps_add(item, item_dict, result_list, owner):
    if item in result_list: return
    dependencies = _get_dependencies(item, item_dict, owner)
    for dependency in dependencies:
        _recursive_deps_add(dependency, item_dict, result_list, owner)
    result_list.append(item)


def _get_dependencies(item, item_dict, owner):
    dependencies = _resolve_dependencies(item.dependencies, item_dict, owner)
    unique_dependencies = set(dependencies)
    assert len(unique_dependencies) == len(dependencies), 'Duplicate dependencies in item %s' % item
    return unique_dependencies


def _resolve_dependencies(dependency, item_dict, owner):
    if dependency is None: return []
    if isinstance(dependency, six.string_types):
        try:
            return [item_dict[dependency]]
        except KeyError:
            raise DependencyError('Declared dependency "%s" does not exist on struct %s. Properties: %s' % (dependency, owner, tuple(item_dict.keys())))
    if isinstance(dependency, Item): return [item_dict[dependency.name]]
    if isinstance(dependency, (tuple, list)):
        return sum([_resolve_dependencies(dep, item_dict, owner) for dep in dependency], [])
    raise ValueError('Cannot resolve dependency "%s". Available items: %s' % (dependency, item_dict.keys()))


class DependencyError(Exception):

    def __init__(self, *args):
        Exception.__init__(self, *args)
