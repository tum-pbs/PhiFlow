from collections import namedtuple

NeighbourReduce = namedtuple('NeighbourReduce', ['requires_weights', 'f'])


def combined_dim(dim1, dim2, type_str: str = 'batch'):
    if dim1 is None and dim2 is None:
        return None
    if dim1 is None or dim1 == 1:
        return dim2
    if dim2 is None or dim2 == 1:
        return dim1
    assert dim1 == dim2, f"Incompatible {type_str} dimensions: x0 {dim1}, y {dim2}"
    return dim1
