from contextlib import contextmanager


_STRUCT_CONTEXT_STACK = []


@contextmanager
def anytype():
    _STRUCT_CONTEXT_STACK.append('anytype')
    try:
        yield None
    finally:
        _STRUCT_CONTEXT_STACK.pop(-1)


def skip_validate():
    return 'anytype' in _STRUCT_CONTEXT_STACK
