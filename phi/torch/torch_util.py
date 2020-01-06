
from phi import math
from .torch_backend import TorchBackend

math.DYNAMIC_BACKEND.add_backend(TorchBackend())
