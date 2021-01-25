import copy

import numpy

from ._trait import Trait


def definition(traits=()):
    """
    Required decorator for custom struct classes.

    Args:
      traits:  (Default value = ())

    Returns:

    """
    if isinstance(traits, Trait):
        traits = (traits,)
    else:
        for trait in traits:
            assert isinstance(trait, Trait), 'Illegal trait: %s' % trait
        traits = tuple(traits)

    def decorator(struct_class, traits=traits):
        assert struct_class.__initialized_class__ != struct_class, 'Struct class already initialized: %s' % struct_class
        items = {}
        for attribute_name in dir(struct_class):
            item = getattr(struct_class, attribute_name)
            if isinstance(item, Item):
                items[attribute_name] = item
        # --- Inheritance ---
        inherited_traits = ()
        for base in struct_class.__bases__:
            if base.__name__ != 'Struct' and hasattr(base, '__items__'):
                for item in base.__items__:
                    if item.name not in items:
                        subclassed_item = copy.copy(item)
                        items[item.name] = subclassed_item
                        setattr(struct_class, item.name, subclassed_item)
                for trait in base.__traits__:
                    if trait not in traits:
                        inherited_traits += (trait,)
        traits = inherited_traits + tuple([t for t in traits if t not in inherited_traits])
        assert len(set(traits)) == len(traits), "Duplicate traits on struct class '%s'" % struct_class
        # --- Initialize & Decorate ---
        struct_class.__traits__ = traits
        for item in items.values():
            item.__initialize_for__(struct_class)
        items = _order_by_dependencies(items, struct_class)
        struct_class.__items__ = tuple(items)
        struct_class.__initialized_class__ = struct_class
        # --- Check trait keywords ---
        for item in items:
            for trait_kw, trait_kw_val in item.trait_kwargs.items():
                matching_traits = [trait for trait in traits if trait_kw in trait.keywords]
                if len(matching_traits) == 0:
                    raise ValueError('Trait keyword "%s" does not match any trait of struct %s' % (trait_kw, struct_class.__name__))
                for trait in matching_traits:
                    trait.check_argument(struct_class, item, trait_kw, trait_kw_val)
        return struct_class
    return decorator


def variable(default=None, dependencies=(), holds_data=True, **trait_kwargs):
    """
    Required decorator for data_dict of custom structs.
    The enclosing class must be decorated with struct.definition().

    Args:
      default: default value passed to validation if no other value is specified
      dependencies: other items (string or reference for inherited constants_dict) (Default value = ())
      holds_data: determines whether the variable is considered by data-related functions (Default value = True)
      **trait_kwargs: 

    Returns:
      read-only property

    """
    def decorator(validate):
        return Item(validate.__name__, validate, True, default, dependencies, holds_data, **trait_kwargs)
    return decorator


def constant(default=None, dependencies=(), holds_data=False, **trait_kwargs):
    """
    Required decorator for constants_dict of custom structs.
    The enclosing class must be decorated with struct.definition().

    Args:
      default: default value passed to validation if no other value is specified
      dependencies: other items (string or reference for inherited constants_dict) (Default value = ())
      holds_data: determines whether the constant is considered by data-related functions (Default value = False)
      **trait_kwargs: 

    Returns:
      read-only property

    """
    def decorator(validate):
        return Item(validate.__name__, validate, False, default, dependencies, holds_data, **trait_kwargs)
    return decorator


def derived():
    """
    Derived properties work similar to @property but can be easily broadcast across many instances.
    :return: read-only property

    Args:

    Returns:

    """
    def decorator(getter):
        return DerivedProperty(getter.__name__, getter)
    return decorator


class Item:
    """Represents an item type of a struct, a variable or a constant."""

    def __init__(self, name, validation_function, is_variable, default_value, dependencies, holds_data, **trait_kwargs):
        assert callable(validation_function) or validation_function is None
        assert isinstance(is_variable, bool)
        self.name = name
        self.validation_function = validation_function
        self.is_variable = is_variable
        self.default_value = default_value
        # --- Format and test dependencies ---
        if dependencies is None:
            self.dependencies = []
        elif isinstance(dependencies, (tuple, list)):
            self.dependencies = list(dependencies)
        else:
            self.dependencies = [dependencies]
        for i, dependency in enumerate(self.dependencies):
            if isinstance(dependency, Item):
                self.dependencies[i] = dependency.name
            elif isinstance(dependency, str):
                pass
            else:
                raise ValueError('Illegal dependency: %s on item %s' % (dependency, name))
        self.dependencies = dependencies
        self.holds_data = holds_data
        self.trait_kwargs = trait_kwargs
        self.struct_class = None
        self._overrides = {}
        self.__doc__ = validation_function.__doc__

    def __initialize_for__(self, struct_class):
        self.struct_class = struct_class
        self.traits = []
        self_kws = list(self.trait_kwargs.keys())
        self.traits = [trait for trait in struct_class.__traits__ if len(numpy.intersect1d(trait.keywords, self_kws)) > 0]

    def set(self, struct, value):
        try:
            setattr(struct, '_' + self.name, value)
        except AttributeError:
            raise AttributeError("can't modify struct %s because item %s cannot be set." % (struct, self))

    def get(self, struct):
        return getattr(struct, '_' + self.name)

    def validate(self, struct):
        if self.validation_function is not None:
            value = self.get(struct)
            for trait in self.traits:
                value = trait.pre_validated(struct, self, value)
            value = self.validation_function(struct, value)
            for trait in self.traits:
                value = trait.post_validated(struct, self, value)
            self.set(struct, value)

    def has_override(self, content_type):
        if content_type is None:
            return False
        return self._attribute_name(content_type) in self._overrides

    def get_override(self, content_type):
        return self._overrides[self._attribute_name(content_type)]

    def override(self, content_type, override_function):
        """
        Override a property or behaviour of this item and/or its values.
        This affects all instances of the associated Struct.
        The override function is called instead of the usual function in `struct.map` to obtain a leaf value.
        
        Overrides can also be used to specify custom property getters, e.g. to override shape, staticshape, dtype.
        As this method is called on an Item, it must be invoked outside the item it affects.
        
        Example: to override the shape of an item, put the following just below its declaration: `item.override(struct.shape, lambda self, value: custom_shape)`

        Args:
          content_type: custom name or Item/DerivedItem reference
          override_function: function, signature depends on the overridden property.

        Returns:

        """
        self._overrides[self._attribute_name(content_type)] = override_function

    @staticmethod
    def _attribute_name(name_or_attribute):
        if isinstance(name_or_attribute, (Item, DerivedProperty)):
            name = name_or_attribute.name
        elif callable(name_or_attribute):
            name = name_or_attribute.__name__
        else:
            name = name_or_attribute
        assert isinstance(name, str), 'Not an attribute: %s' % name
        return name

    def __get__(self, instance, owner):
        if instance is not None:
            return getattr(instance, '_' + self.name)
        else:
            return self

    def __call__(self, obj):
        assert self.struct_class is not None
        from ._struct_functions import map
        return map(lambda x: getattr(x, '_' + self.name), obj, leaf_condition=lambda x: isinstance(x, self.struct_class))

    def __set__(self, instance, value):
        raise AttributeError('Struct variables and constants are read-only.')

    def __delete__(self, instance):
        raise AttributeError('Struct variables and constants are read-only.')

    def __repr__(self):
        return self.name


class _IndexItem(Item):

    def __init__(self, index, is_variable=True, holds_data=True):
        Item.__init__(self, name=index, validation_function=None, is_variable=is_variable, default_value=None, dependencies=(), holds_data=holds_data)
        self.index = index

    def get(self, struct):
        return struct[self.index]

    def set(self, struct, value):
        struct[self.index] = value


class DerivedProperty:

    def __init__(self, name, getter):
        self.name = name
        self.getter = getter
        self.__doc__ = getter.__doc__

    def __get__(self, instance, owner):
        if instance is not None:
            return self.getter(instance)
        else:
            self.owner = owner
            return self

    def __call__(self, obj):
        assert self.owner is not None
        from ._struct_functions import map
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
    if item in result_list:
        return
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
    if dependency is None:
        return []
    if isinstance(dependency, str):
        try:
            return [item_dict[dependency]]
        except KeyError:
            raise DependencyError('Declared dependency "%s" does not exist on struct %s. Properties: %s' % (dependency, struct_cls.__name__, tuple(item_dict.keys())))
    if isinstance(dependency, Item):
        return [item_dict[dependency.name]]
    if isinstance(dependency, (tuple, list)):
        return sum([_resolve_dependencies(dep, item_dict, struct_cls) for dep in dependency], [])
    raise ValueError('Cannot resolve dependency "%s". Available items: %s' % (dependency, item_dict.keys()))


class DependencyError(Exception):

    def __init__(self, *args):
        Exception.__init__(self, *args)
