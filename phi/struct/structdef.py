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


def attr(default=None, dependencies=()):
    """
Required decorator for attributes of custom structs.
The enclosing class must be decorated with struct.definition().
    :param default: default value passed to validation if no other value is specified
    :param dependencies: other items (string or reference for inherited properties)
    :return: read-only attribute
    """
    def decorator(validate):
        item = Item(validate.__name__, validate, True, default, dependencies)
        _register_item(validate, item)
        return item.get_property()
    return decorator


def prop(default=None, dependencies=()):
    """
Required decorator for properties of custom structs.
The enclosing class must be decorated with struct.definition().
    :param default: default value passed to validation if no other value is specified
    :param dependencies: other items (string or reference for inherited properties)
    :return: read-only property
    """
    def decorator(validate):
        item = Item(validate.__name__, validate, False, default, dependencies)
        _register_item(validate, item)
        return item.get_property()
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
        self.items = _order_by_dependencies(item_dict)
        self.attributes = tuple(filter(lambda item: item.is_attribute, self.items))
        self.properties = tuple(filter(lambda item: not item.is_attribute, self.items))

    def find(self, name):
        return self.item_dict[name]

    def validate(self, struct):
        for item in self.items:
            item.validate(struct)

    @property
    def item_names(self):
        return [item.name for item in self.items]


class Item(object):
    """
Represents an item type of a struct, an attribute or property.
    """

    def __init__(self, name, validation_function, is_attribute, default_value, dependencies):
        assert isinstance(name, six.string_types)
        assert callable(validation_function)
        self.name = name
        self.validation_function = validation_function
        self.is_attribute = is_attribute
        self.default_value = default_value
        self.dependencies = dependencies
        self.unique_property = property(lambda struct: getattr(struct, '_'+self.name))

    def set(self, struct, value):
        try:
            setattr(struct, '_' + self.name, value)
        except AttributeError:
            raise AttributeError("can't modify struct %s because attribute %s cannot be set." % (struct, self))

    def get(self, struct):
        return getattr(struct, '_' + self.name)

    def get_property(self):
        return self.unique_property

    def validate(self, struct):
        old_val = self.get(struct)
        new_val = self.validation_function(struct, old_val)
        self.set(struct, new_val)

    def __repr__(self):
        return self.name


def _order_by_dependencies(item_dict):
    result = []
    for item in item_dict.values():
        _recursive_deps_add(item, item_dict, result)
    return result


def _recursive_deps_add(item, item_dict, result_list):
    if item in result_list: return
    dependencies = _get_dependencies(item, item_dict)
    for dependency in dependencies:
        _recursive_deps_add(dependency, item_dict, result_list)
    result_list.append(item)


def _get_dependencies(item, item_dict):
    dependencies = _resolve_dependencies(item.dependencies, item_dict)
    unique_dependencies = set(dependencies)
    assert len(unique_dependencies) == len(dependencies), 'Duplicate dependencies in item %s' % item
    return unique_dependencies


def _resolve_dependencies(dependency, item_dict):
    if dependency is None: return []
    if isinstance(dependency, six.string_types): return [item_dict[dependency]]
    if isinstance(dependency, property):
        for item in item_dict.values():
            if item.unique_property == dependency:
                return [item]
    if isinstance(dependency, (tuple, list)):
        return sum([_resolve_dependencies(dep, item_dict) for dep in dependency], [])
    raise ValueError('Cannot resolve dependency %s. Available items: %s' % (dependency, item_dict.keys()))