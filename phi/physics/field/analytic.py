from phi.backend.backend import Backend
from phi import math, struct
from phi.geom.geometry import assert_same_rank

from .field import Field


@struct.definition()
class AnalyticField(Field):

    def __init__(self, rank, data=None, name=None, **kwargs):
        Field.__init__(self, **struct.kwargs(locals(), ignore='rank'))
        self._rank = rank

    @property
    def rank(self):
        return self._rank

    @property
    def component_count(self):
        if self._rank is None:
            return None
        try:
            example_value = self.sample_at([[0] * self._rank])
            return math.staticshape(example_value)[-1]
        except NotImplementedError:
            return None

    def unstack(self):
        assert self.component_count is not None, 'The component_count of %s is unknown' % self
        if self.component_count == 1:
            return [self]
        else:
            components = []
            for i in range(self.component_count):
                def _context(index=i):
                    return lambda x: x.unstack()[index]
                components.append(_SymbolicOpField(_context(i), [self]))
            return components

    @property
    def points(self):
        return None

    def compatible(self, other_field):
        return True

    def __mul__(self, other):
        return _SymbolicOpField(lambda x1, x2: x1 * x2, [self, other])

    __rmul__ = __mul__

    def __div__(self, other):
        return _SymbolicOpField(lambda x1, x2: x1 / x2, [self, other])

    def __truediv__(self, other):
        return _SymbolicOpField(lambda x1, x2: x1 / x2, [self, other])

    def __sub__(self, other):
        return _SymbolicOpField(lambda x1, x2: x1 - x2, [self, other])

    def __rsub__(self, other):
        return _SymbolicOpField(lambda x1, x2: x2 - x1, [self, other])

    def __add__(self, other):
        return _SymbolicOpField(lambda x1, x2: x1 + x2, [self, other])

    __radd__ = __add__

    def __pow__(self, power, modulo=None):
        return _SymbolicOpField(lambda x1, x2: x1 ** x2, [self, power])

    def __dataop__(self, other, linear_if_scalar, data_operator):
        return _SymbolicOpField(data_operator, [self, other])

    @struct.constant(default=None)
    def data(self, data):
        return data


class SymbolicFieldBackend(Backend):
    # Abstract mehtods are overridden generically.
    # pylint: disable-msg = abstract-method

    def __init__(self, backend):
        Backend.__init__(self, 'Symbolic Field Backend')
        self.backend = backend
        for fname in dir(self):
            if fname not in ('__init__', 'is_tensor', 'is_applicable', 'symbolic_call') and not fname.startswith('__'):
                function = getattr(self, fname)
                if callable(function):
                    def context(fname=fname):
                        def proxy(*args, **kwargs):
                            return self.symbolic_call(fname, args, kwargs)
                        return proxy
                    setattr(self, fname, context())

    def symbolic_call(self, func, args, kwargs):
        assert len(kwargs) == 0, 'kwargs not supported'
        backend_func = getattr(self.backend, func)
        return _SymbolicOpField(backend_func, args)

    def is_tensor(self, x, only_native=False):
        return isinstance(x, AnalyticField)


@struct.definition()
class _SymbolicOpField(AnalyticField):

    def __init__(self, function, function_args, **kwargs):
        fields = filter(lambda arg: isinstance(arg, Field), function_args)
        AnalyticField.__init__(self, _determine_rank(fields), name=function.__name__, **struct.kwargs(locals(), ignore='fields'))
        self.fields = tuple(fields)
        self.channels = _determine_component_count(function_args)

    @struct.constant()
    def function_args(self, args):
        return args

    @struct.constant()
    def function(self, function):
        return function

    def at(self, other_field):
        args = []
        for arg in self.function_args:
            if isinstance(arg, Field):
                arg = arg.at(other_field)
            args.append(arg)
        applied = self.function(*args)
        return applied

    def sample_at(self, points):
        raise NotImplementedError()

    @property
    def component_count(self):
        return self.channels

    def unstack(self):
        unstacked = {}
        for arg in self.function_args:
            if isinstance(arg, Field):
                unstacked[arg] = arg.unstack()
            elif math.is_tensor(arg) and math.ndims(arg) > 0:
                unstacked[arg] = math.unstack(arg, axis=-1, keepdims=True)
            else:
                unstacked[arg] = [arg] * self.component_count
            assert len(unstacked[arg]) == self.component_count
        result = [_SymbolicOpField(self.function, [unstacked[arg][i] for arg in self.function_args]) for i in range(self.component_count)]
        return result

    @property
    def points(self):
        if len(self.fields) == 0:
            return None
        else:
            return self.fields[0].points

    def compatible(self, other_field):
        if len(self.fields) == 0:
            return True
        else:
            return self.fields[0].compatible(other_field)


def _determine_rank(fields):
    rank = None
    for field in fields:
        if rank is None:
            rank = field.rank
        else:
            assert_same_rank(rank, field.rank, 'All fields must have the same rank')
    return rank


def _determine_component_count(args):
    result = None
    for arg in args:
        arg_channels = None
        if isinstance(arg, Field):
            arg_channels = arg.component_count
        elif math.is_tensor(arg) and math.ndims(arg) > 0:
            arg_channels = arg.shape[-1]
        if result is None:
            result = arg_channels
        elif result == 1 and arg_channels is not None:
            result = arg_channels
        else:
            assert result == arg_channels or arg_channels is None
    return result
