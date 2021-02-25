import warnings
from contextlib import contextmanager


_STRUCT_CONTEXT_STACK = []


@contextmanager
def _struct_context(object):
    _STRUCT_CONTEXT_STACK.append(object)
    try:
        yield None
    finally:
        _STRUCT_CONTEXT_STACK.pop(-1)


def unsafe():
    warnings.warn("struct.unsafe() is deprecated. Use map() with new_type argument to avoid validation.")
    return _struct_context('unsafe')


def _unsafe():
    return _struct_context('unsafe')


def skip_validate():
    return 'unsafe' in _STRUCT_CONTEXT_STACK
