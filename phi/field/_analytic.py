from abc import ABC

from phi.math.backend import Backend
from phi import math
from phi.math import Shape, Tensor

from ._field import Field, SampledField
from ..geom import Geometry


class AnalyticField(Field):

    def unstack(self, dimension: str) -> tuple:
        components = []
        size = self.shape.get_size(dimension)
        shape = self.shape.without(dimension)
        for i in range(size):
            def _context(index=i):
                return lambda x: x.unstack(dimension)[index]
            components.append(_SymbolicOpField(shape, _context(i), [self]))
        return tuple(components)

    def _op2(self, other, operator):
        if isinstance(other, SampledField):
            self_sampled = self.at(other)
            data = operator(self_sampled.values, other.values)
            return other.with_(values=data)
        other = math.tensor(other)
        new_shape = self.shape.combined(other.shape)
        return _SymbolicOpField(new_shape, operator, [self, other])

    def _op1(self, operator):
        return _SymbolicOpField(self.shape, operator, [self])


class _SymbolicOpField(AnalyticField):

    def __init__(self, shape, function, function_args):
        self.function = function
        self.function_args = function_args
        fields = filter(lambda arg: isinstance(arg, Field), function_args)
        self.fields = tuple(fields)
        self._shape = shape

    @property
    def shape(self) -> Shape:
        return self._shape

    def sample_in(self, geometry: Geometry, reduce_channels=()) -> Tensor:
        args = []
        for arg in self.function_args:
            if isinstance(arg, Field):
                arg = arg.sample_in(geometry, reduce_channels=reduce_channels)
            args.append(arg)
        applied = self.function(*args)
        return applied

    def sample_at(self, points, reduce_channels=()) -> math.Tensor:
        args = []
        for arg in self.function_args:
            if isinstance(arg, Field):
                arg = arg.sample_at(points, reduce_channels=reduce_channels)
            args.append(arg)
        applied = self.function(*args)
        return applied

    def unstack(self, dimension):
        unstacked = {}
        for arg in self.function_args:
            if isinstance(arg, Field):
                unstacked[arg] = arg.unstack()
            elif math.is_tensor(arg) and math.ndims(arg) > 0:
                unstacked[arg] = math.unstack(arg, dim=-1, keepdims=True)
            else:
                unstacked[arg] = [arg] * self.component_count
            assert len(unstacked[arg]) == self.component_count
        result = [_SymbolicOpField(self.function, [unstacked[arg][i] for arg in self.function_args]) for i in range(self.component_count)]
        return result

