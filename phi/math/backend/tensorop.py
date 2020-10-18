import numpy as np


def _is_leaf(tensor_like, leaf_condition):
    if not isinstance(tensor_like, (tuple, list, np.ndarray)):
        return True
    if leaf_condition is not None and leaf_condition(tensor_like):
        return True
    return False


def collapse(tensor_like, leaf_condition=None):
    if _is_leaf(tensor_like, leaf_condition):
        return tensor_like
    collapsed_elements = tuple([collapse(element, leaf_condition) for element in tensor_like])
    first = collapsed_elements[0]
    for element in collapsed_elements[1:]:
        if element != first:
            return collapsed_elements
    return first


def collapsed_gather_nd(collapsed, nd_index, leaf_condition=None):
    if isinstance(collapsed, (tuple, list, np.ndarray)):
        if leaf_condition is not None and leaf_condition(collapsed):
            return collapsed
        # collapsed = np.array(collapsed)
        if len(nd_index) == 1:
            return collapsed[nd_index[0]]
        else:
            return collapsed_gather_nd(collapsed[nd_index[0]], nd_index[1:])
    else:
        return collapsed


def expand(collapsed, shape):
    if len(shape) == 0:
        return collapsed
    if isinstance(collapsed, (tuple, list, np.ndarray)):
        if len(collapsed) == shape[0]:
            return [expand(item, shape[1:]) for item in collapsed]
        elif len(collapsed) == 1:
            item = expand(collapsed[0], shape[1:])
            return [item] * shape[0]
        else:
            raise ValueError('Cannot match shape: requested %d but actual %d' % (shape[0], len(collapsed)))
    else:
        return [expand(collapsed, shape[1:])] * shape[0]


class CollapsedTensor(object):

    def __init__(self, collapsed, leaf_condition=None, shape=None):
        self.collapsed = collapsed
        self.leaf_condition = leaf_condition
        self.shape = shape

    def __getitem__(self, item):
        return collapsed_gather_nd(self.collapsed, item, self.leaf_condition)

    def expand(self, shape=None):
        shape = self.shape if shape is None else shape
        assert shape is not None
        return expand(self.collapsed, shape)
