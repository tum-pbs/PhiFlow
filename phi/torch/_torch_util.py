import torch

from phi import struct
from ._torch_backend import TORCH_BACKEND


def variable(initial_value, dtype=None, requires_grad=True, device=None, item_condition=struct.VARIABLES):
    def f(attr): return torch.tensor(attr.value, dtype=dtype if dtype is not None else TORCH_BACKEND.precision_dtype, requires_grad=requires_grad, device=device)
    return struct.map(f, initial_value, trace=True, item_condition=item_condition)


def torch_from_numpy(obj, item_condition=struct.DATA):
    return struct.map(torch.from_numpy, obj, item_condition=item_condition)


def torch_to_numpy(obj, item_condition=struct.ALL_ITEMS):
    def to_numpy(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numpy()
        else:
            return obj
    return struct.map(to_numpy, obj, item_condition=item_condition)
