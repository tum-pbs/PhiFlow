
class Struct(object):

    def disassemble(self):
        """
Disassembles the container into its element tensors.
    :return: 1. List of component tensors, 2. Reassemble: A function that recombines these tensors to produce an equivalent TensorContainer
        """
        raise NotImplementedError(self)


def shape(struct):
    tensors, reassemble = disassemble(struct)
    shapes = [tensor.shape for tensor in tensors]
    return reassemble(shapes)


def disassemble(struct):
    if isinstance(struct, Struct):
        return struct.disassemble()
    if isinstance(struct, (tuple, list)):
        return _disassemble_list(struct)
    else:
        return [struct], lambda tensors: tensors[0]


def _disassemble_list(struct_list):
    tensor_counts = []
    reassemblers = []
    tensor_list = []
    for struct in struct_list:
        tensors, reassemble = disassemble(struct)
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


def gather_batches(struct, batches):
    if isinstance(batches, int):
        batches = [batches]
    tensors, reassemble = disassemble(struct)
    new_tensors = [tensor[batches,...] for tensor in tensors]
    return reassemble(new_tensors)