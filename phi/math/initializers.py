from .struct import *
from .nd import upsample2x
from .base import backend as math
import numpy as np


def zeros(shape, dtype=np.float32):
    shapes, reassemble = disassemble(shape)
    zeros = [np.zeros(shape, dtype) for shape in shapes]
    return reassemble(zeros)

def zeros_like(object):
    tensors, reassemble = disassemble(object)
    zeros = [math.zeros_like(tensor) for tensor in tensors]
    return reassemble(zeros)

def ones(shape, dtype=np.float32):
    shapes, reassemble = disassemble(shape)
    zeros = [np.ones(shape, dtype) for shape in shapes]
    return reassemble(zeros)

def ones_like(object):
    tensors, reassemble = disassemble(object)
    zeros = [np.ones_like(tensor) for tensor in tensors]
    return reassemble(zeros)

def empty(shape, dtype=np.float32):
    shapes, reassemble = disassemble(shape)
    zeros = [np.empty(shape, dtype) for shape in shapes]
    return reassemble(zeros)

def empty_like(object):
    tensors, reassemble = disassemble(object)
    zeros = [np.empty_like(tensor) for tensor in tensors]
    return reassemble(zeros)

def randn(shape, levels=(1.0,)):
    shapes, reassemble = disassemble(shape)
    zeros = [_random_tensor(shape, levels) for shape in shapes]
    return reassemble(zeros)

def _random_tensor(shape, levels):
    result = 0
    for i in range(len(levels)): # high-res first
        lowres_shape = np.array(shape)
        lowres_shape[1:-1] //= 2 ** i
        rnd = np.random.randn(*lowres_shape) * levels[i]
        for j in range(i):
            rnd = upsample2x(rnd)
        result = result + rnd
    return result