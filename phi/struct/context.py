from contextlib import contextmanager


_stack = []


@contextmanager
def anytype():
    _stack.append('anytype')
    try:
        yield None
    finally:
        _stack.pop(-1)


def skip_validate():
    return 'anytype' in _stack


print(skip_validate())
with anytype():
    print(skip_validate())
print(skip_validate())