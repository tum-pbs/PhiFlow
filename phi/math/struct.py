from copy import copy
import numpy as np


class StructAttr(object):

    def __init__(self, ref_string, name=None, is_property=False):
        self.ref_string = ref_string
        self.name = name if name is not None else (ref_string[1:] if ref_string.startswith('_') else ref_string)
        self.is_property = is_property

    def matches(self, string):
        return self.name == string or self.ref_string == string

    def __repr__(self):
        return self.ref_string


class StructInfo(object):

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
        return StructInfo(self.attributes + list(attributes), self.properties + list(properties))


class Struct(object):
    __struct__ = StructInfo(())

    def copied_with(self, **kwargs):
        duplicate = copy(self)
        for name, value in kwargs.items():
            attr = self.__class__.__struct__.find(name).ref_string
            setattr(duplicate, attr, value)
        return duplicate

    def __values__(self):
        return [getattr(self, a.name) for a in self.__class__.__struct__.attributes]

    def __properties__(self):
        result = {p.name: Struct.properties(getattr(self, p.name)) for p in self.__class__.__struct__.properties}
        result['type'] = str(self.__class__.__name__)
        result['module'] = str(self.__class__.__module__)
        return result

    def __names__(self):
        return self.__class__.__struct__.attributes

    def __mapnames__(self, basename):
        values = self.__values__()
        result_list = []
        for attr, value in zip(self.__class__.__struct__.attributes, values):
            fullname = attr.name if basename is None else basename+'.'+attr.name
            if Struct.isstruct(value):
                result_list.append(Struct.mapnames(value, fullname))
            else:
                result_list.append(fullname)
        return self.__build__(result_list)

    def __build__(self, values):
        new_struct = copy(self)
        for val, attr in zip(values, self.__class__.__struct__.attributes):
            setattr(new_struct, attr.ref_string, val)
        return new_struct

    def __flatten__(self):
        values = self.__values__()
        flat_list, recombine = _flatten_list(values)
        def recombine_self(flat_list):
            values = recombine(flat_list)
            return self.__build__(values)
        return flat_list, recombine_self

    def __eq__(self, other):
        if type(self) != type(other): return False
        for attr in self.__class__.__struct__.all:
            v1 = getattr(self, attr.name)
            v2 = getattr(other, attr.name)
            try:
                if v1 != v2: return False
            except:
                if v1 is not v2: return False
        return True

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(self.__values__()))

    @staticmethod
    def values(struct):
        if isinstance(struct, Struct):
            return struct.__values__()
        if isinstance(struct, (list, tuple)):
            return struct
        if isinstance(struct, dict):
            return struct.values()
        raise ValueError("Not a struct: %s" % struct)

    @staticmethod
    def properties(struct):
        if isinstance(struct, Struct):
            return struct.__properties__()
        if isinstance(struct, (list, tuple)):
            return [Struct.properties(s) for s in struct]
        if isinstance(struct, dict):
            return {key: Struct.properties(value) for key,value in struct.items()}
        if isinstance(struct, np.ndarray):
            assert len(struct.shape) == 1
            struct = struct.tolist()
        import json, io
        try:
            json.dump(struct, io.BytesIO())
            return struct
        except:
            raise TypeError('Object "%s" of type "%s" is not JSON serializable' % (struct,type(struct)))

    @staticmethod
    def names(struct):
        if isinstance(struct, Struct):
            return struct.__names__()
        if isinstance(struct, (list, tuple)):
            return ['[%d]' % i for i in range(len(struct))]
        if isinstance(struct, dict):
            return struct.keys()
        raise ValueError("Not a struct: %s" % struct)

    @staticmethod
    def mapnames(struct, basename=None):
        if isinstance(struct, Struct):
            return struct.__mapnames__(basename)
        if isinstance(struct, (list, tuple)):
            return _mapnames_list(struct, basename)
        if isinstance(struct, dict):
            return _mapnames_dict(struct, basename)
        raise ValueError("Not a struct: %s" % struct)

    @staticmethod
    def build(values, source_struct):
        if isinstance(source_struct, Struct):
            return source_struct.__build__(values)
        if isinstance(source_struct, list):
            return list(values)
        if isinstance(source_struct, tuple):
            return tuple(values)
        if isinstance(source_struct, dict):
            return {name: value for value, (name, _) in zip(values, source_struct.items())}
        raise ValueError("Not a struct: %s" % source_struct)

    @staticmethod
    def flatten(struct):
        if isinstance(struct, Struct):
            return struct.__flatten__()
        if isinstance(struct, (tuple, list)):
            return _flatten_list(struct)
        if isinstance(struct, dict):
            return _flatten_dict(struct)
        return [struct], lambda tensors: tensors[0]

    @staticmethod
    def isstruct(object):
        return isinstance(object, (Struct, list, tuple, dict))

    @staticmethod
    def map(f, struct):
        values = Struct.values(struct)
        values = [f(element) for element in values]
        return Struct.build(values, struct)

    @staticmethod
    def zippedmap(f, struct, *structs):
        main_values = Struct.values(struct)
        others_values = [Struct.values(s) for s in structs]
        result_values = []
        for i in range(len(main_values)):
            result_value = f(main_values[i], *[values[i] for values in others_values])
            result_values.append(result_value)
        return Struct.build(result_values, struct)

    @staticmethod
    def zippedflatmap(f, struct, *structs):
        main_values, main_recombine = Struct.flatten(struct)
        others_values = [Struct.flatten(s)[0] for s in structs]
        result_values = []
        for i in range(len(main_values)):
            result_value = f(main_values[i], *[values[i] for values in others_values])
            result_values.append(result_value)
        return main_recombine(result_values)


    @staticmethod
    def flatmap(f, struct):
        values, recombine = Struct.flatten(struct)
        values = [f(element) for element in values]
        return recombine(values)



def _flatten_list(struct_list):
    tensor_counts = []
    recombiners = []
    values = []
    for struct in struct_list:
        tensors, recombine = Struct.flatten(struct)
        values += tensors
        tensor_counts.append(len(tensors))
        recombiners.append(recombine)

    def recombine(tensor_list):
        new_structs = []
        for i in range(len(struct_list)):
            tensors = tensor_list[:tensor_counts[i]]
            struct = recombiners[i](tensors)
            new_structs.append(struct)
            tensor_list = tensor_list[tensor_counts[i]:]
        assert len(tensor_list) == 0, "Not all tensors were used in reassembly"
        if isinstance(struct_list, list):
            return new_structs
        else:
            return tuple(new_structs)

    return values, recombine


def _flatten_dict(struct_dict):
    values = struct_dict.values()
    keys = struct_dict.keys()

    values_list, recombine_list = _flatten_list(values)

    def recombine(values_list):
        values = recombine_list(values_list)
        return {key: value for key, value in zip(keys, values)}

    return values_list, recombine


def _mapnames_list(struct, basename):
    fullname = lambda i: "%d" % i if basename is None else basename + '.%d' % i
    result = []
    for i, element in enumerate(struct):
        if Struct.isstruct(element):
            result.append(Struct.mapnames(element, fullname(i)))
        else:
            result.append(fullname(i))
    return tuple(result) if isinstance(struct, tuple) else result


def _mapnames_dict(struct, basename):
    fullname = lambda key: key if basename is None else basename + '.' + key
    result = {}
    for key, value in struct.items():
        if Struct.isstruct(value):
            result[key] = Struct.mapnames(value, fullname(key))
        else:
            result[key] = fullname(key)
    return result
