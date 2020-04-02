from phi import math
from .torch_backend import TorchBackend

TORCH_BACKEND = TorchBackend()
math.DYNAMIC_BACKEND.add_backend(TORCH_BACKEND)
