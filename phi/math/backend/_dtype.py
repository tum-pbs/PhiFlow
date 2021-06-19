import numpy as np


class DType:
    """
    Instances of `DType` represent the kind and size of data elements.
    The data type of a `Tensor` can be obtained via `phi.math.Tensor.dtype`.

    The following kinds of data types are supported:

    * `float` with 32 / 64 bits
    * `complex` with 64 / 128 bits
    * `int` with 8 / 16 / 32 / 64 bits
    * `bool` with 8 bits
    * `str` with 8*n* bits

    Unlike with many computing libraries, there are no global variables corresponding to the available types.
    Instead, data types can simply be instantiated as needed.
    """

    def __init__(self, kind: type, bits: int = 8):
        """
        Args:
            kind: Python type, one of `(bool, int, float, complex, str)`
            bits: number of bits per element, a multiple of 8.
        """
        assert kind in (bool, int, float, complex, str)
        if kind is bool:
            assert bits == 8
        else:
            assert isinstance(bits, int)
        self.kind = kind
        """ Python class corresponding to the type of data, ignoring precision. One of (bool, int, float, complex) """
        self.bits = bits
        """ Number of bits used to store a single value of this type. See `DType.itemsize`. """

    @property
    def precision(self):
        """ Floating point precision. Only defined if `kind in (float, complex)`. For complex values, returns half of `DType.bits`. """
        if self.kind == float:
            return self.bits
        if self.kind == complex:
            return self.bits // 2
        else:
            return None

    @property
    def itemsize(self):
        """ Number of bytes used to storea single value of this type. See `DType.bits`. """
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
    if dtype in _TO_NUMPY:
        return _TO_NUMPY[dtype]
    if dtype.kind == str:
        bytes_per_char = np.dtype('<U1').itemsize
        return np.dtype(f'<U{dtype.itemsize // bytes_per_char}')


def from_numpy_dtype(np_dtype) -> DType:
    if np_dtype in _FROM_NUMPY:
        return _FROM_NUMPY[np_dtype]
    else:
        for base_np_dtype, dtype in _FROM_NUMPY.items():
            if np_dtype == base_np_dtype:
                return dtype
        if np_dtype.char == 'U':
            return DType(str, 8 * np_dtype.itemsize)
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


def combine_types(*dtypes: DType, fp_precision: int) -> DType:
    # all bool?
    if all(dt.kind == bool for dt in dtypes):
        return dtypes[0]
    # all int / bool?
    if all(dt.kind in (bool, int) for dt in dtypes):
        largest = max(dtypes, key=lambda dt: dt.bits)
        return largest
    # all real?
    if all(dt.kind in (float, int, bool) for dt in dtypes):
        return DType(float, fp_precision)
    # complex
    if all(dt.kind in (complex, float, int, bool) for dt in dtypes):
        return DType(complex, 2 * fp_precision)
    # string
    if any(dt.kind == str for dt in dtypes):
        largest = max([dt for dt in dtypes if dt.kind == str], key=lambda dt: dt.bits)
        return largest
    raise ValueError(dtypes)
