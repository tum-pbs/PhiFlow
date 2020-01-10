import numpy as np

import torch

from phi import math, struct
from .torch_backend import TorchBackend

math.DYNAMIC_BACKEND.add_backend(TorchBackend())


def variable(initial_value, dtype=torch.float32, requires_grad=True, device=None, item_condition=struct.VARIABLES):
    def f(attr): return torch.tensor(attr.value, dtype=dtype, requires_grad=requires_grad, device=device)
    return struct.map(f, initial_value, trace=True, item_condition=item_condition)