from contextlib import contextmanager


_STRUCT_CONTEXT_STACK = []


@contextmanager
def unsafe():
    _STRUCT_CONTEXT_STACK.append('unsafe')
    try:
        yield None
    finally:
        _STRUCT_CONTEXT_STACK.pop(-1)


def skip_validate():
    return 'unsafe' in _STRUCT_CONTEXT_STACK
