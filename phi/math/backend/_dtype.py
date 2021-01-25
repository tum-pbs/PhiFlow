import numpy as np


class DType:

    def __init__(self, kind, bits: int = 8):
        """
        Data type for tensors.

        Args:
          kind: Python type, one of bool, int, float, complex
          bits: number of bits, typically a multiple of 8.
        """
        assert kind in (bool, int, float, complex)
        if kind is bool:
            assert bits == 8
        else:
            assert isinstance(bits, int)
        self.kind = kind
        self.bits = bits

    @property
    def precision(self):
        if self.kind == float:
            return self.bits
        if self.kind == complex:
            return self.bits // 2
        else:
            return None

    @property
    def itemsize(self):
        assert self.bits % 8 == 0
        return self.bits // 8

    def __eq__(self, other):
        return isinstance(other, DType) and self.kind == other.kind and self.bits == other.bits

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.kind) + hash(self.bits)

    def __repr__(self):
        return f"{self.kind.__name__}{self.bits}"


# --- NumPy Conversion ---

def to_numpy_dtype(dtype: DType):
    return _TO_NUMPY[dtype]


def from_numpy_dtype(np_dtype):
    if np_dtype in _FROM_NUMPY:
        return _FROM_NUMPY[np_dtype]
    else:
        for base_np_dtype, dtype in _FROM_NUMPY.items():
            if np_dtype == base_np_dtype:
                return dtype
        raise ValueError(np_dtype)


_TO_NUMPY = {
    DType(float, 16): np.float16,
    DType(float, 32): np.float32,
    DType(float, 64): np.float64,
    DType(complex, 64): np.complex64,
    DType(complex, 128): np.complex128,
    DType(int, 8): np.int8,
    DType(int, 16): np.int16,
    DType(int, 32): np.int32,
    DType(int, 64): np.int64,
    DType(bool): np.bool_,
}
_FROM_NUMPY = {np: dtype for dtype, np in _TO_NUMPY.items()}
_FROM_NUMPY[np.bool] = DType(bool)
