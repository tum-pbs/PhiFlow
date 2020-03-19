from phi.backend.backend import Backend
from . import context, struct, functions


class StructBroadcastBackend(Backend):
    # Abstract mehtods are overridden generically.
    # pylint: disable-msg = abstract-method

    def __init__(self, backend, target_content_type=struct.VALID):
        Backend.__init__(self, 'StructBroadcast')
        self.backend = backend
        self.target_content_type = target_content_type
        for fname in dir(self):
            if fname not in ('__init__', 'is_applicable', 'broadcast_function') and not fname.startswith('__'):
                function = getattr(self, fname)
                if callable(function):
                    def context(fname=fname):
                        def proxy(*args, **kwargs):
                            return self.broadcast_function(self.backend, fname, args, kwargs)
                        return proxy
                    setattr(self, fname, context())

    def is_applicable(self, values):
        for value in values:
            if struct.isstruct(value):
                return True
        return False

    def broadcast_function(self, backend, func, args, kwargs):
        backend_func = getattr(backend, func)
        obj, build_arguments = argument_assembler(args, kwargs)

        def f(*values):
            args, kwargs = build_arguments(values)
            result = backend_func(*args, **kwargs)
            return result
        target_content_type = {
            'staticshape': functions.staticshape,
            'shape': functions.shape,
            'dtype': functions.dtype,
        }.get(func, self.target_content_type)
        return functions.map(f, obj, content_type=target_content_type)


def argument_assembler(args, kwargs):
    structs, keymap = build_keymap(args, kwargs)
    assert len(structs) > 0
    if len(structs) == 1:
        obj = structs[0]
    else:
        obj = functions.zip(structs)

    def assemble_arguments(items):
        args = []
        kwargs = {}
        i = 0
        while i in keymap:
            is_struct, value = keymap[i]
            if is_struct:
                value = items[value]
            args.append(value)
            i += 1
        for key, (is_struct, value) in keymap.items():
            if not isinstance(key, int):
                if is_struct:
                    value = items[value]
                kwargs[key] = value
        return args, kwargs

    return obj, assemble_arguments


def build_keymap(args, kwargs):
    " maps key to (False, value) or (True, key); keys are integers for args and strings for kwargs "
    map = {}
    structs = []
    for i, value in enumerate(args):
        if struct.isstruct(value):
            map[i] = (True, len(structs))
            structs.append(value)
        else:
            map[i] = (False, value)
    for key, value in kwargs.items():
        if struct.isstruct(value):
            map[key] = (True, len(structs))
            structs.append(value)
        else:
            map[key] = (False, value)
    return structs, map
