from phi.math.base import Backend
from phi import struct


class StructBroadcastBackend(Backend):

    def __init__(self, backend):
        Backend.__init__(self, 'StructBroadcast')
        self.backend = backend
        for fname in dir(self):
            if fname not in ('__init__', 'is_applicable', 'broadcast_function') and not fname.startswith('__'):
                function = getattr(self, fname)
                if callable(function):
                    def context(fname=fname):
                        def proxy(*args, **kwargs):
                            return self.broadcast_function(args[0], fname, *args[1:], **kwargs)
                        return proxy
                    setattr(self, fname, context())

    def is_applicable(self, values):
        for value in values:
            if struct.isstruct(value): return True
        return False

    def broadcast_function(self, obj, func, *args, **kwargs):
        backend_func = getattr(self.backend, func)
        f = lambda x: backend_func(x, *args, **kwargs)
        return struct.map(f, obj)

