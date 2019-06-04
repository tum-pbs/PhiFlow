

class TensorContainer(object):

    def disassemble(self):
        """
Disassembles the container into its element tensors.
    :return: 1. List of component tensors, 2. Reassemble: A function that recombines these tensors to produce an equivalent TensorContainer
        """
        raise NotImplementedError(self)


def disassemble(tensor_or_container):
    if isinstance(tensor_or_container, TensorContainer):
        return tensor_or_container.disassemble()
    else:
        return [tensor_or_container], lambda tensors: tensors[0]


def list_tensors(containers):
    tensor_counts = []
    reassemblers = []
    tensor_list = []
    for container in containers:
        tensors, reassemble = disassemble(container)
        tensor_list += tensors
        tensor_counts.append(len(tensors))
        reassemblers.append(reassemble)

    def reassemble(tensor_list):
        new_containers = []
        for i in range(len(containers)):
            tensors = tensor_list[:tensor_counts[i]]
            tensorcontainer = reassemblers[i](tensors)
            new_containers.append(tensorcontainer)
            tensor_list = tensor_list[tensor_counts[i]:]
        assert len(tensor_list) == 0, "Not all tensors were used in reassembly"
        return new_containers

    return tensor_list, reassemble
