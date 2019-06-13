from copy import copy


class StructInfo(object):

    def __init__(self, attributes):
        self.attributes = tuple(attributes)


class Struct(object):
    __struct__ = StructInfo(())

    def __disassemble__(self):
        """
Disassembles the container into its element tensors.
    :return: 1. List of component tensors, 2. Reassemble: A function that recombines these tensors to produce an equivalent TensorContainer
        """
        def reassemble(values):
            new_struct = copy(self)
            for val, attr in zip(values, self.__class__.__struct__.attributes):
                setattr(new_struct, attr, val)
            return new_struct

        return Struct.values(self), reassemble

    @staticmethod
    def foreach(struct, element_function):
        tensors, reassemble = Struct.disassemble(struct)
        tensors = [element_function(element) for element in tensors]
        return reassemble(tensors)

    @staticmethod
    def values(struct):
        if isinstance(struct, (list, tuple)):
            return struct
        if isinstance(struct, Struct):
            attributes = struct.__class__.__struct__.attributes
            values = [getattr(struct, a) for a in attributes]
            return values
        raise ValueError("Not a struct: %s" % struct)

    @staticmethod
    def disassemble(struct):
        if isinstance(struct, Struct):
            return struct.__disassemble__()
        if isinstance(struct, (tuple, list)):
            return _flatten_list(struct)
        else:
            raise ValueError("Not a struct: %s" % struct)
            return [struct], lambda tensors: tensors[0]

    @staticmethod
    def isstruct(object):
        return isinstance(object, (Struct, list, tuple))

    @staticmethod
    def flatten(struct):
        if isinstance(struct, Struct):
            return struct.__disassemble__()
        if isinstance(struct, (tuple, list)):
            return _flatten_list(struct)
        else:
            return [struct], lambda tensors: tensors[0]


def _flatten_list(struct_list):
    tensor_counts = []
    reassemblers = []
    tensor_list = []
    for struct in struct_list:
        tensors, reassemble = Struct.disassemble(struct)
        tensor_list += tensors
        tensor_counts.append(len(tensors))
        reassemblers.append(reassemble)

    def reassemble(tensor_list):
        new_structs = []
        for i in range(len(struct_list)):
            tensors = tensor_list[:tensor_counts[i]]
            struct = reassemblers[i](tensors)
            new_structs.append(struct)
            tensor_list = tensor_list[tensor_counts[i]:]
        assert len(tensor_list) == 0, "Not all tensors were used in reassembly"
        return new_structs

    return tensor_list, reassemble


def shape(struct):
    return Struct.foreach(struct, lambda tensor: tensor.shape)


def batch_gather(struct, batches):
    if isinstance(batches, int):
        batches = [batches]
    return Struct.foreach(struct, lambda tensor: tensor[batches,...])


def attributes(struct, remove_prefix=True, qualified_names=True):
    array, reassemble = Struct.disassemble(struct)
    ids = ["id%d" % i for i in range(len(array))]
    id_struct = reassemble(ids)
    _recursive_attributes(id_struct, ids, remove_prefix, qualified_names, None)
    return id_struct


def _recursive_attributes(struct, ids, remove_prefix, qualified_names, qualifier):
    if not Struct.isstruct(struct): return

    if isinstance(struct, (tuple,list)):
        for entry in struct:
            _recursive_attributes(entry, ids, remove_prefix, qualified_names, qualifier)
    else:  # must be a Struct instance
        for attr, val in struct.__dict__.items():
            name = attr
            if remove_prefix and name.startswith('_'):
                name = name[1:]
            if qualified_names:
                if qualifier is None: qualified_name = name
                else: qualified_name = qualifier + '.' + name
            else:
                qualified_name = name
            if val in ids:
                setattr(struct, attr, qualified_name)
            else:
                _recursive_attributes(val, ids, remove_prefix, qualified_names, qualified_name)


# class StructAttributeGetter(object):
#
#     def __init__(self, getter):
#         self.getter = getter
#
#     def __call__(self, struct):
#         return self.getter(struct)
#
#
# def selector(struct):
#     array, reassemble = disassemble(struct)
#     ids = ['#ref' % i for i in range(len(array))]
#     tagged_struct = reassemble(ids)
#     _recursive_selector(tagged_struct)
#     return tagged_struct
#
#
# def _recursive_selector(struct):
#     pass
