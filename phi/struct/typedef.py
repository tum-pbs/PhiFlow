import inspect, six
from .collection import Batch


_struct_register = {}  # map from (module, name) to StructType


def register_item_by_function(function, item):
    classname = function.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
    module = inspect.getmodule(function)
    register_item_by_id(module, classname, item)


def register_item_by_id(module, struct_classname, item):
    class_key = (module, struct_classname)
    if class_key not in _struct_register:
        _struct_register[class_key] = StructType(module, struct_classname)
    _struct_register[class_key].items.append(item)


def get_type(struct_class):
    """
    :rtype: StructType
    """
    module = inspect.getmodule(struct_class)
    classname = struct_class.__name__
    key = (module, classname)
    if key not in _struct_register:
        from .struct import Struct
        assert issubclass(struct_class, Struct), 'Not a Struct: %s' % struct_class.__name__
        structtype = StructType(module, classname)
        _struct_register[key] = structtype
    else:
        structtype = _struct_register[key]
    if not structtype.bases_added:
        structtype.add_bases(struct_class.__bases__)
    return structtype


class StructType(object):

    def __init__(self, module, struct_classname):
        self.module = module
        self.struct_classname = struct_classname
        self.items = []
        self.bases_added = False

    @property
    def attributes(self):
        return filter(lambda item: item.is_attribute, self.items)

    @property
    def properties(self):
        return filter(lambda item: not item.is_attribute, self.items)

    def find(self, name):
        for item in self.items:
            if item.name == name:
                return item
        raise KeyError('Attribute "%s" not found on struct %s' % (name, self.struct_classname))

    def validate(self, struct):
        for item in self.items:
            item.validate(struct)

    def add_bases(self, bases):
        from .struct import Struct
        for base in bases:
            if issubclass(base, Struct) and base != Struct:
                t = get_type(base)
                for item in t.items:
                    if item.name not in self.item_names:
                        self.items.append(item)
        self.bases_added = True

    @property
    def item_names(self):
        return [item.name for item in self.items]


class Item(object):

    def __init__(self, name, validation_function, is_attribute, default_value, innate_dimensions):  # , stack_behavior
        assert isinstance(name, six.string_types)
        assert callable(validation_function)
        assert isinstance(innate_dimensions, int) or innate_dimensions is None
        self.name = name
        self.validation_function = validation_function
        self.is_attribute = is_attribute
        self.default_value = default_value
        self.innate_dimensions = innate_dimensions
        # self.stack_behavior = stack_behavior

    def set(self, struct, value):
        try:
            setattr(struct, '_' + self.name, value)
        except AttributeError as e:
            raise AttributeError("can't modify struct %s because attribute %s cannot be set." % (struct, self))

    def get(self, struct):
        return getattr(struct, '_' + self.name)

    def property(self):
        return property(lambda struct: getattr(struct, '_'+self.name))

    def validate(self, struct):
        old_val = self.get(struct)
        # TODO add dimensions (maybe through dependency injection
        # if self.innate_dimensions is not None:
        #     dims = tensor_rank(old_val)
        #     if dims < self.innate_dimensions:
        #         dims = expand_dims(old_val, self.innate_dimensions - dims)
        new_val = self.validation_function(struct, old_val)
        self.set(struct, new_val)

    def __repr__(self):
        return self.name


def tensor_rank(tensor):
    return len(tensor.shape)


def expand_dims(tensor, number_dims):
    assert number_dims >= 0
    if number_dims == 0: return tensor
    else:
        tensor = Batch([tensor])
        return expand_dims(tensor, number_dims-1)