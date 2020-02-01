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
    return _struct_context('unsafe')


def skip_validate():
    return 'unsafe' in _STRUCT_CONTEXT_STACK
