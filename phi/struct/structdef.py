import typing  # pylint: disable-msg = unused-import  # this is used in # type
from typing import Dict
import six


def definition():
    """
Required decorator for custom struct classes.
    """
    def decorator(cls):
        items = {}
        for attribute_name in dir(cls):
            item = getattr(cls, attribute_name)
            if isinstance(item, Item):
                items[attribute_name] = item
        # --- Inherit items ---
        for base in cls.__bases__:
            if base.__name__ != 'Struct' and hasattr(base, '__items__'):
                for item in base.__items__:
                    if item.name not in items:
                        items[item.name] = item
        # --- Order items by dependencies ---
        items = _order_by_dependencies(items, cls)
        cls.__items__ = tuple(items)
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


class Item(object):
    """
Represents an item type of a struct, a variable or a constant.
    """

    def __init__(self, name, validation_function, is_variable, default_value, dependencies, holds_data, **trait_kwargs):
        assert callable(validation_function) or validation_function is None
        self.name = name
        self.validation_function = validation_function
        self.is_variable = is_variable
        self.default_value = default_value
        self.dependencies = dependencies
        self.holds_data = holds_data
        self.trait_kwargs = trait_kwargs
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


def _order_by_dependencies(item_dict, struct_cls):
    result = []
    for item in item_dict.values():
        _recursive_deps_add(item, item_dict, result, struct_cls)
    return result


def _recursive_deps_add(item, item_dict, result_list, struct_cls):
    if item in result_list: return
    dependencies = _get_dependencies(item, item_dict, struct_cls)
    for dependency in dependencies:
        _recursive_deps_add(dependency, item_dict, result_list, struct_cls)
    result_list.append(item)


def _get_dependencies(item, item_dict, struct_cls):
    dependencies = _resolve_dependencies(item.dependencies, item_dict, struct_cls)
    unique_dependencies = set(dependencies)
    assert len(unique_dependencies) == len(dependencies), 'Duplicate dependencies in item %s' % item
    return unique_dependencies


def _resolve_dependencies(dependency, item_dict, struct_cls):
    if dependency is None: return []
    if isinstance(dependency, six.string_types):
        try:
            return [item_dict[dependency]]
        except KeyError:
            raise DependencyError('Declared dependency "%s" does not exist on struct %s. Properties: %s' % (dependency, struct_cls.__name__, tuple(item_dict.keys())))
    if isinstance(dependency, Item): return [item_dict[dependency.name]]
    if isinstance(dependency, (tuple, list)):
        return sum([_resolve_dependencies(dep, item_dict, struct_cls) for dep in dependency], [])
    raise ValueError('Cannot resolve dependency "%s". Available items: %s' % (dependency, item_dict.keys()))


class DependencyError(Exception):

    def __init__(self, *args):
        Exception.__init__(self, *args)
